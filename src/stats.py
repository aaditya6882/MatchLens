import numpy as np
import pandas as pd
from config import FPS
from geometry import attack_x

MAX_GAP_S = 0.1          # larger time gaps between frames are ignored
MAX_SPEED = 12.0         # m/s: faster is a tracking glitch
SMOOTH_FRAMES = 12       # 0.5 s rolling mean
SPRINT_SPEED = 7.0       # m/s (about 25 km/h)
SPRINT_MIN_S = 1.0


def compute_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Players only, with step_m (metres since previous frame) and speed_ms (smoothed)."""
    p = df[df.team != "ball"].sort_values(["player_id", "period", "frame"]).copy()
    g = p.groupby(["player_id", "period"], sort=False)
    dt = g.time_s.diff()
    step = np.hypot(g.x.diff(), g.y.diff())
    ok = (dt > 0) & (dt <= MAX_GAP_S)
    raw = (step / dt).where(ok)
    raw = raw.where(raw <= MAX_SPEED)                    # glitches -> NaN
    p["step_m"] = step.where(raw.notna())
    p["speed_ms"] = raw.groupby([p.player_id, p.period]).transform(
        lambda s: s.rolling(SMOOTH_FRAMES, min_periods=SMOOTH_FRAMES // 2, center=True).mean())
    return p


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs in a boolean array."""
    padded = np.concatenate(([0], mask.astype(int), [0]))
    d = np.diff(padded)
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def player_stats(df: pd.DataFrame) -> pd.DataFrame:
    p = compute_speed(df)
    rows = []
    for pid, g in p.groupby("player_id"):
        sprints = 0
        for _, gp in g.groupby("period"):
            fast = (gp.speed_ms.fillna(0) > SPRINT_SPEED).to_numpy()
            sprints += int((_run_lengths(fast) >= SPRINT_MIN_S * FPS).sum())
        rows.append({
            "player_id": pid,
            "team": g.team.iloc[0],
            "minutes": round(len(g) / FPS / 60, 1),
            "distance_km": round(g.step_m.sum() / 1000, 2),
            "top_speed_kmh": round(g.speed_ms.max() * 3.6, 1),
            "sprints": sprints,
        })
    return pd.DataFrame(rows).sort_values(["team", "player_id"]).reset_index(drop=True)


def team_shape(df: pd.DataFrame, team: str, every: int = 25) -> pd.DataFrame:
    """Width, depth, defensive line height once per `every` frames (25 = once a second).
    Needs a normalized df. Attacking direction, goalkeeper excluded."""
    t = df[(df.team == team) & (df.frame % every == 0)].copy()
    t["ax"] = attack_x(t)
    rows = []
    for (frame, period, time_s), g in t.groupby(["frame", "period", "time_s"]):
        outfield = g.sort_values("ax").iloc[1:]          # drop the deepest player (keeper)
        if len(outfield) < 8:
            continue
        rows.append({
            "frame": frame, "period": period, "time_s": time_s,
            "width_m": outfield.y.max() - outfield.y.min(),
            "depth_m": outfield.ax.max() - outfield.ax.min(),
            "def_line_x": outfield.ax.iloc[:4].mean(),   # 4 deepest outfield players
        })
    return pd.DataFrame(rows)