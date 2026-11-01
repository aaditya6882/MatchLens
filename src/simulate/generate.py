"""Synthetic match generator producing tracking tables and ground-truth events."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import random
import sys
from pathlib import Path

# Support running directly as a script: python src/sim/generate.py
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.config import DATA_DIR
from src.events.schema import Event

FPS = 25
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

HALF_LENGTH = PITCH_LENGTH / 2.0  # 52.5
HALF_WIDTH = PITCH_WIDTH / 2.0    # 34.0
ATTACK_THIRD_X = HALF_LENGTH / 3.0  # 17.5 m from center; final third is |x| >= 17.5

# Standard 4-3-3 base formations (home attacks towards +x, away towards -x)
HOME_FORMATION: Dict[str, Tuple[float, float]] = {
    "home_1": (-48.0, 0.0),    # GK
    "home_2": (-30.0, -22.0),  # RB
    "home_3": (-32.0, -8.0),   # CB
    "home_4": (-32.0, 8.0),    # CB
    "home_5": (-30.0, 22.0),   # LB
    "home_6": (-15.0, 0.0),    # DM
    "home_7": (-5.0, -14.0),   # CM
    "home_8": (-5.0, 14.0),    # CM
    "home_9": (15.0, -20.0),   # RW
    "home_10": (20.0, 0.0),    # ST
    "home_11": (15.0, 20.0),   # LW
}

AWAY_FORMATION: Dict[str, Tuple[float, float]] = {
    "away_1": (48.0, 0.0),     # GK
    "away_2": (30.0, 22.0),    # RB
    "away_3": (32.0, 8.0),     # CB
    "away_4": (32.0, -8.0),    # CB
    "away_5": (30.0, -22.0),   # LB
    "away_6": (15.0, 0.0),     # DM
    "away_7": (5.0, 14.0),     # CM
    "away_8": (5.0, -14.0),    # CM
    "away_9": (-15.0, 20.0),   # RW
    "away_10": (-20.0, 0.0),   # ST
    "away_11": (-15.0, -20.0),  # LW
}


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _lerp(p1: Tuple[float, float], p2: Tuple[float, float], t: float) -> Tuple[float, float]:
    return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)


@dataclass
class MatchData:
    match_id: str
    tracking_df: pd.DataFrame
    truth_events: List[Event]
    truth_df: pd.DataFrame


class MatchSimulator:
    def __init__(self, match_id: str = "m1", duration_s: float = 180.0, fps: int = FPS, seed: Optional[int] = 42):
        self.match_id = match_id
        self.duration_s = duration_s
        self.fps = fps
        self.total_frames = int(duration_s * fps)
        self.rng = random.Random(seed)

        # Player states
        self.player_teams: Dict[str, str] = {}
        self.base_positions: Dict[str, Tuple[float, float]] = {}
        self.positions: Dict[str, Tuple[float, float]] = {}

        for pid, pos in HOME_FORMATION.items():
            self.player_teams[pid] = "home"
            self.base_positions[pid] = pos
            self.positions[pid] = pos

        for pid, pos in AWAY_FORMATION.items():
            self.player_teams[pid] = "away"
            self.base_positions[pid] = pos
            self.positions[pid] = pos

        # Ball state
        self.ball_pos: Tuple[float, float] = (0.0, 0.0)
        self.possessor: Optional[str] = "home_6"
        self.current_team: str = "home"

        # Pass / shot motion
        self.ball_in_flight: bool = False
        self.ball_start: Tuple[float, float] = (0.0, 0.0)
        self.ball_target: Tuple[float, float] = (0.0, 0.0)
        self.ball_flight_start_f: int = 0
        self.ball_flight_end_f: int = 0
        self.ball_receiver: Optional[str] = None
        self.is_accelerated_shot: bool = False

        # Phase timing
        self.phase_end_f: int = 0
        self.current_phase: str = "open_play"
        self.phase_data: dict = {}

        # Logged ground truth events
        self.truth_events: List[Event] = []

    def _schedule_scripted_events(self) -> List[dict]:
        """Plan scripted events (shot, attack, key_passage) evenly distributed across duration."""
        # We need at least one of each event type
        event_types = ["shot", "attack", "key_passage"]
        # If match is long enough, repeat or shuffle
        planned = []
        # Calculate time windows: leave 15s margins at start and end
        usable_start = 15.0
        usable_end = max(usable_start + 30.0, self.duration_s - 15.0)
        
        # Determine number of events (at least 3, approx 1 event every 35-50s)
        n_events = max(3, int(self.duration_s / 40.0))
        sequence = (event_types * ((n_events // 3) + 1))[:n_events]
        self.rng.shuffle(sequence)

        slot_duration = (usable_end - usable_start) / n_events
        for idx, ev_type in enumerate(sequence):
            slot_start = usable_start + idx * slot_duration
            start_s = slot_start + self.rng.uniform(2.0, max(2.5, slot_duration - 12.0))
            team = "home" if self.rng.random() < 0.5 else "away"
            planned.append({
                "type": ev_type,
                "team": team,
                "start_s": start_s,
            })
        return sorted(planned, key=lambda x: x["start_s"])

    def _teammates(self, team: str) -> List[str]:
        return [p for p, t in self.player_teams.items() if t == team]

    def _pick_pass_target(self, passer: str, forward_bias: float = 0.0) -> str:
        team = self.player_teams[passer]
        candidates = [p for p in self._teammates(team) if p != passer and not p.endswith("_1")]
        curr_x, curr_y = self.positions[passer]
        direction = 1.0 if team == "home" else -1.0

        scored = []
        for c in candidates:
            cx, cy = self.positions[c]
            dist = _dist((curr_x, curr_y), (cx, cy))
            if 6.0 <= dist <= 38.0:
                fwd = (cx - curr_x) * direction
                score = 1.0 + forward_bias * (fwd / 20.0)
                scored.append((c, max(0.1, score)))

        if not scored:
            return self.rng.choice(candidates)
        total = sum(s for _, s in scored)
        r = self.rng.uniform(0, total)
        cum = 0.0
        for c, s in scored:
            cum += s
            if cum >= r:
                return c
        return scored[-1][0]

    def _start_pass(self, receiver: str, current_frame: int, speed_mps: float = 14.0):
        self.ball_in_flight = True
        self.ball_start = self.ball_pos
        self.ball_target = self.positions[receiver]
        self.ball_receiver = receiver
        self.possessor = None
        self.is_accelerated_shot = False

        distance = _dist(self.ball_start, self.ball_target)
        duration_s = max(0.4, distance / speed_mps)
        self.ball_flight_start_f = current_frame
        self.ball_flight_end_f = current_frame + int(duration_s * self.fps)

    def _start_shot(self, shooter: str, current_frame: int):
        team = self.player_teams[shooter]
        goal_x = HALF_LENGTH if team == "home" else -HALF_LENGTH
        goal_y = self.rng.uniform(-4.0, 4.0)

        self.ball_in_flight = True
        self.ball_start = self.ball_pos
        self.ball_target = (goal_x, goal_y)
        self.ball_receiver = None
        self.possessor = None
        self.is_accelerated_shot = True

        distance = _dist(self.ball_start, self.ball_target)
        # Fast shot: 24 to 30 m/s
        duration_s = max(0.5, distance / 26.0)
        self.ball_flight_start_f = current_frame
        self.ball_flight_end_f = current_frame + int(duration_s * self.fps)

    def run(self) -> MatchData:
        scripted_events = self._schedule_scripted_events()
        next_event_idx = 0

        records = []

        for frame in range(self.total_frames):
            time_s = frame / self.fps

            # Check if it's time to start the next scripted event
            if next_event_idx < len(scripted_events):
                next_ev = scripted_events[next_event_idx]
                if time_s >= next_ev["start_s"] and self.current_phase == "open_play" and not self.ball_in_flight:
                    ev_type = next_ev["type"]
                    ev_team = next_ev["team"]
                    next_event_idx += 1

                    if ev_type == "shot":
                        # Set up attacking third shot
                        self.current_phase = "shot"
                        direction = 1.0 if ev_team == "home" else -1.0
                        self.current_team = ev_team
                        # Pick forward/midfielder
                        attackers = [p for p in self._teammates(ev_team) if p.split("_")[1] in ("7", "8", "9", "10", "11")]
                        shooter = self.rng.choice(attackers)
                        self.possessor = shooter
                        # Move shooter into attacking third (x between 20 and 38 in attacking dir)
                        sx = direction * self.rng.uniform(22.0, 36.0)
                        sy = self.rng.uniform(-16.0, 16.0)
                        self.positions[shooter] = (sx, sy)
                        self.ball_pos = (sx, sy)

                        start_time = time_s
                        shot_flight_dur = _dist((sx, sy), (direction * HALF_LENGTH, 0.0)) / 26.0
                        end_time = round(start_time + shot_flight_dur, 2)
                        self.truth_events.append(Event(
                            match_id=self.match_id,
                            type="shot",
                            team=ev_team,
                            start_s=round(start_time, 2),
                            end_s=end_time,
                            source="ground_truth",
                            status="confirmed",
                            note="Scripted shot from attacking third",
                        ))
                        self._start_shot(shooter, frame)
                        self.phase_end_f = self.ball_flight_end_f + int(1.5 * self.fps)

                    elif ev_type == "attack":
                        # Keep ball in final third for several seconds (e.g. 5-7s)
                        attack_dur = self.rng.uniform(5.0, 7.5)
                        self.current_phase = "attack"
                        self.current_team = ev_team
                        self.phase_end_f = frame + int(attack_dur * self.fps)
                        start_time = time_s
                        end_time = round(start_time + attack_dur, 2)
                        self.truth_events.append(Event(
                            match_id=self.match_id,
                            type="attack",
                            team=ev_team,
                            start_s=round(start_time, 2),
                            end_s=end_time,
                            source="ground_truth",
                            status="confirmed",
                            note="Scripted sustained attack in final third",
                        ))
                        # Teleport/bias possessor into attacking third
                        direction = 1.0 if ev_team == "home" else -1.0
                        poss = self.rng.choice([p for p in self._teammates(ev_team) if not p.endswith("_1")])
                        self.possessor = poss
                        self.positions[poss] = (direction * self.rng.uniform(24.0, 42.0), self.rng.uniform(-20.0, 20.0))
                        self.ball_pos = self.positions[poss]

                    elif ev_type == "key_passage":
                        # Chain of 3 fast passes with large forward gain
                        self.current_phase = "key_passage"
                        self.current_team = ev_team
                        kp_dur = 6.0
                        self.phase_end_f = frame + int(kp_dur * self.fps)
                        start_time = time_s
                        end_time = round(start_time + kp_dur, 2)
                        self.truth_events.append(Event(
                            match_id=self.match_id,
                            type="key_passage",
                            team=ev_team,
                            start_s=round(start_time, 2),
                            end_s=end_time,
                            source="ground_truth",
                            status="confirmed",
                            note="Scripted key passage with forward progression",
                        ))
                        direction = 1.0 if ev_team == "home" else -1.0
                        # Start in defensive/mid half and push forward
                        mid = self.rng.choice([p for p in self._teammates(ev_team) if p.split("_")[1] in ("2", "3", "4", "5", "6")])
                        self.possessor = mid
                        self.positions[mid] = (direction * self.rng.uniform(-25.0, -10.0), self.rng.uniform(-15.0, 15.0))
                        self.ball_pos = self.positions[mid]
                        self.phase_data = {"passes_left": 3, "next_pass_f": frame}

            # Update ball motion
            if self.ball_in_flight:
                total_flight = max(1, self.ball_flight_end_f - self.ball_flight_start_f)
                progress = (frame - self.ball_flight_start_f) / total_flight
                progress = _clamp(progress, 0.0, 1.0)

                if self.is_accelerated_shot:
                    # Accelerating movement: t^1.5 or quadratic
                    ease_progress = math.pow(progress, 1.4)
                else:
                    ease_progress = progress

                self.ball_pos = _lerp(self.ball_start, self.ball_target, ease_progress)

                if frame >= self.ball_flight_end_f:
                    self.ball_in_flight = False
                    if self.ball_receiver:
                        self.possessor = self.ball_receiver
                        self.ball_pos = self.positions[self.possessor]
                    else:
                        # Shot concluded: goalkeeper or defending team recovers ball
                        defending_team = "away" if self.current_team == "home" else "home"
                        keeper = f"{defending_team}_1"
                        self.possessor = keeper
                        self.current_team = defending_team
                        self.ball_pos = self.positions[keeper]

            # Update phase logic
            if self.current_phase == "shot":
                if frame >= self.phase_end_f and not self.ball_in_flight:
                    self.current_phase = "open_play"

            elif self.current_phase == "attack":
                direction = 1.0 if self.current_team == "home" else -1.0
                # Keep players and ball in final third (x >= 17.5 in attack direction)
                if self.possessor and not self.ball_in_flight:
                    # Pass occasionally to another teammate in attacking third
                    if self.rng.random() < 0.08:
                        cands = [p for p in self._teammates(self.current_team) if p != self.possessor and not p.endswith("_1")]
                        recv = self.rng.choice(cands)
                        # Ensure receiver is in final third
                        rx = direction * self.rng.uniform(20.0, 44.0)
                        ry = self.rng.uniform(-24.0, 24.0)
                        self.positions[recv] = (rx, ry)
                        self._start_pass(recv, frame, speed_mps=15.0)

                if frame >= self.phase_end_f:
                    self.current_phase = "open_play"

            elif self.current_phase == "key_passage":
                direction = 1.0 if self.current_team == "home" else -1.0
                if self.phase_data.get("passes_left", 0) > 0 and not self.ball_in_flight:
                    if frame >= self.phase_data.get("next_pass_f", 0):
                        curr_p = self.possessor
                        if curr_p:
                            cx, cy = self.positions[curr_p]
                            target_p = self._pick_pass_target(curr_p, forward_bias=3.0)
                            # Move target forward
                            tx = cx + direction * self.rng.uniform(14.0, 22.0)
                            tx = _clamp(tx, -HALF_LENGTH + 5.0, HALF_LENGTH - 5.0)
                            ty = self.rng.uniform(-20.0, 20.0)
                            self.positions[target_p] = (tx, ty)
                            self._start_pass(target_p, frame, speed_mps=18.0)
                            self.phase_data["passes_left"] -= 1
                            self.phase_data["next_pass_f"] = self.ball_flight_end_f + int(0.4 * self.fps)

                if frame >= self.phase_end_f and not self.ball_in_flight:
                    self.current_phase = "open_play"

            elif self.current_phase == "open_play":
                # Regular open play passing
                if self.possessor and not self.ball_in_flight:
                    # Occasional pass or turnover
                    if self.rng.random() < 0.04:  # roughly every second
                        if self.rng.random() < 0.15:
                            # Turnover / interception
                            other_team = "away" if self.current_team == "home" else "home"
                            recv = self.rng.choice([p for p in self._teammates(other_team) if not p.endswith("_1")])
                            self.current_team = other_team
                            self._start_pass(recv, frame, speed_mps=12.0)
                        else:
                            recv = self._pick_pass_target(self.possessor, forward_bias=0.5)
                            self._start_pass(recv, frame, speed_mps=14.0)

            # Update player positions (formation drift + shifting with ball)
            ball_x, ball_y = self.ball_pos
            shift_x = ball_x * 0.25  # team lines squeeze & shift up/down with ball
            shift_y = ball_y * 0.15

            for pid, (bx, by) in self.base_positions.items():
                if pid == self.possessor:
                    # Possessor stays right at ball
                    self.positions[pid] = (ball_x, ball_y)
                    continue

                if self.ball_in_flight and pid == self.ball_receiver:
                    # Receiver stays near target
                    continue

                is_gk = pid.endswith("_1")
                gk_factor = 0.15 if is_gk else 1.0

                # Target home position shifted by ball position
                target_x = bx + shift_x * gk_factor
                target_y = by + shift_y * gk_factor

                # Small random drift
                drift_x = self.rng.uniform(-0.15, 0.15)
                drift_y = self.rng.uniform(-0.15, 0.15)

                px, py = self.positions[pid]
                # Smooth movement toward target + drift
                new_x = px + (target_x - px) * 0.04 + drift_x
                new_y = py + (target_y - py) * 0.04 + drift_y

                # Keep players strictly within pitch boundaries
                new_x = _clamp(new_x, -HALF_LENGTH + 1.0, HALF_LENGTH - 1.0)
                new_y = _clamp(new_y, -HALF_WIDTH + 1.0, HALF_WIDTH - 1.0)
                self.positions[pid] = (new_x, new_y)

            # Record frame data: each player + ball
            for pid, (px, py) in self.positions.items():
                records.append({
                    "frame": frame,
                    "time_s": round(time_s, 2),
                    "team": self.player_teams[pid],
                    "player_id": pid,
                    "x": round(px, 3),
                    "y": round(py, 3),
                })

            records.append({
                "frame": frame,
                "time_s": round(time_s, 2),
                "team": "ball",
                "player_id": "ball",
                "x": round(self.ball_pos[0], 3),
                "y": round(self.ball_pos[1], 3),
            })

        tracking_df = pd.DataFrame.from_records(records)
        truth_records = [e.to_dict() for e in self.truth_events]
        truth_columns = ["match_id", "type", "team", "start_s", "end_s", "confidence", "source", "status", "note", "id"]
        truth_df = pd.DataFrame(truth_records, columns=truth_columns)

        return MatchData(
            match_id=self.match_id,
            tracking_df=tracking_df,
            truth_events=self.truth_events,
            truth_df=truth_df,
        )


def generate_match(
    match_id: str = "m1",
    duration_s: float = 180.0,
    fps: int = FPS,
    seed: Optional[int] = 42,
    output_dir: Optional[Path] = None,
) -> MatchData:
    """Generates synthetic match tracking and ground-truth events, saving CSV files automatically."""
    sim = MatchSimulator(match_id=match_id, duration_s=duration_s, fps=fps, seed=seed)
    data = sim.run()

    if output_dir is None:
        output_dir = DATA_DIR / "matches" / match_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_path = output_dir / "tracking.csv"
    truth_path = output_dir / "truth.csv"

    data.tracking_df.to_csv(tracking_path, index=False)
    data.truth_df.to_csv(truth_path, index=False)

    return data


def plot_frame(df: pd.DataFrame, frame: int, title: str = "", save_path: Optional[Path] = None):
    """Plot a single frame of tracking data on a plausible pitch."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle, Rectangle

    f = df[df.frame == frame]
    if f.empty:
        raise ValueError(f"Frame {frame} not found in tracking DataFrame.")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_facecolor("#2e7d32")  # Pitch grass green

    # Pitch boundary
    ax.add_patch(Rectangle((-HALF_LENGTH, -HALF_WIDTH), PITCH_LENGTH, PITCH_WIDTH,
                           fill=False, edgecolor="white", linewidth=2))

    # Halfway line & center circle
    ax.plot([0, 0], [-HALF_WIDTH, HALF_WIDTH], color="white", linewidth=1.5)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, edgecolor="white", linewidth=1.5))
    ax.plot(0, 0, marker="o", color="white", markersize=3)

    # Penalty areas (16.5m deep, 40.3m wide)
    pa_w, pa_h = 16.5, 40.32
    ax.add_patch(Rectangle((-HALF_LENGTH, -pa_h / 2), pa_w, pa_h, fill=False, edgecolor="white", linewidth=1.5))
    ax.add_patch(Rectangle((HALF_LENGTH - pa_w, -pa_h / 2), pa_w, pa_h, fill=False, edgecolor="white", linewidth=1.5))

    # Goal areas (5.5m deep, 18.32m wide)
    ga_w, ga_h = 5.5, 18.32
    ax.add_patch(Rectangle((-HALF_LENGTH, -ga_h / 2), ga_w, ga_h, fill=False, edgecolor="white", linewidth=1.5))
    ax.add_patch(Rectangle((HALF_LENGTH - ga_w, -ga_h / 2), ga_w, ga_h, fill=False, edgecolor="white", linewidth=1.5))

    # Penalty spots & penalty arcs
    ax.plot(-HALF_LENGTH + 11.0, 0, marker="o", color="white", markersize=3)
    ax.plot(HALF_LENGTH - 11.0, 0, marker="o", color="white", markersize=3)
    ax.add_patch(Arc((-HALF_LENGTH + 11.0, 0), 18.3, 18.3, angle=0, theta1=307, theta2=53, color="white", linewidth=1.5))
    ax.add_patch(Arc((HALF_LENGTH - 11.0, 0), 18.3, 18.3, angle=0, theta1=127, theta2=233, color="white", linewidth=1.5))

    # Goals
    gw = 7.32
    ax.plot([-HALF_LENGTH, -HALF_LENGTH - 2], [-gw / 2, -gw / 2], color="white", linewidth=2)
    ax.plot([-HALF_LENGTH, -HALF_LENGTH - 2], [gw / 2, gw / 2], color="white", linewidth=2)
    ax.plot([-HALF_LENGTH - 2, -HALF_LENGTH - 2], [-gw / 2, gw / 2], color="white", linewidth=2)

    ax.plot([HALF_LENGTH, HALF_LENGTH + 2], [-gw / 2, -gw / 2], color="white", linewidth=2)
    ax.plot([HALF_LENGTH, HALF_LENGTH + 2], [gw / 2, gw / 2], color="white", linewidth=2)
    ax.plot([HALF_LENGTH + 2, HALF_LENGTH + 2], [-gw / 2, gw / 2], color="white", linewidth=2)

    # Plot Home team
    home = f[f.team == "home"]
    ax.scatter(home.x, home.y, c="#d32f2f", s=130, edgecolors="white", linewidth=1.5, label="Home", zorder=4)
    for _, row in home.iterrows():
        num = row.player_id.split("_")[1]
        ax.text(row.x, row.y, num, color="white", fontsize=8, ha="center", va="center", fontweight="bold", zorder=5)

    # Plot Away team
    away = f[f.team == "away"]
    ax.scatter(away.x, away.y, c="#1565c0", s=130, edgecolors="white", linewidth=1.5, label="Away", zorder=4)
    for _, row in away.iterrows():
        num = row.player_id.split("_")[1]
        ax.text(row.x, row.y, num, color="white", fontsize=8, ha="center", va="center", fontweight="bold", zorder=5)

    # Plot Ball
    ball = f[f.team == "ball"]
    if not ball.empty:
        bx, by = ball.iloc[0].x, ball.iloc[0].y
        ax.scatter(bx, by, c="#ffeb3b", s=90, edgecolors="black", linewidth=1.5, label="Ball", zorder=6)

    ax.set_xlim(-HALF_LENGTH - 4, HALF_LENGTH + 4)
    ax.set_ylim(-HALF_WIDTH - 4, HALF_WIDTH + 4)
    ax.set_aspect("equal")
    ax.axis("off")

    time_s = f.iloc[0].time_s
    ax.set_title(title or f"Match Frame {frame} (t = {time_s:.2f}s)", color="black", fontsize=13, pad=10)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


if __name__ == "__main__":
    data = generate_match("m1", duration_s=60.0, seed=42)
    print(f"Generated match m1: {len(data.tracking_df)} tracking rows, {len(data.truth_events)} events.")
    print("Events:")
    for ev in data.truth_events:
        print(f"  {ev.type:12s} | {ev.team:4s} | {ev.start_s:6.2f}s -> {ev.end_s:6.2f}s | {ev.note}")
