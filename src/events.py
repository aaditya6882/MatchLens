import numpy as np
import pandas as pd
from loader import load_tracking
from geometry import normalize_pitch

FINAL_THIRD_X = 17.5     # attacking third starts 35 m from the goal line (52.5 - 35)
POSSESSION_DIST = 3.0    # m: nearest player within this distance has the ball
HOLD_S = 3.0             # keep possession through short gaps (ball in the air)
MIN_SPELL_S = 2.0


def possession_frames(df: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    """One row per `step` frames: ball position and which team has it (or NaN)."""
    ball = df[df.team == "ball"].set_index("frame")[["x", "y", "period", "time_s"]]
    pl = df[(df.team != "ball") & (df.frame % step == 0)]
    m = pl.merge(ball[["x", "y"]], left_on="frame", right_index=True, suffixes=("", "_b"))
    m["d"] = np.hypot(m.x - m.x_b, m.y - m.y_b)
    near = m.loc[m.groupby("frame").d.idxmin()].set_index("frame")
    f = ball.loc[near.index].copy()
    f["team"] = np.where(near.d < POSSESSION_DIST, near.team, None)
    f["team"] = f["team"].ffill(limit=int(HOLD_S * 25 / step))
    return f.reset_index()


def possession_spells(f: pd.DataFrame) -> pd.DataFrame:
    f = f.dropna(subset=["team"]).copy()
    new = (f.team != f.team.shift()) | (f.period != f.period.shift()) | (f.time_s.diff() > HOLD_S)
    f["spell"] = new.cumsum()
    return f


def detect_events(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_pitch(df)
    f = possession_spells(possession_frames(df))
    f["att_x"] = np.where(f.team == "home", f.x, -f.x)   # progress towards the team's goal
    events, prev_team = [], None
    for _, s in f.groupby("spell"):
        team, period = s.team.iloc[0], int(s.period.iloc[0])
        t0, t1 = s.time_s.iloc[0], s.time_s.iloc[-1]
        if t1 - t0 < MIN_SPELL_S:
            continue
        start_x = s.att_x.iloc[0]
        advance = s.att_x.max() - start_x

        # 1) attacking sequence: 3-45 s of possession, ball progressed 20+ m
        if 3 <= t1 - t0 <= 45 and advance >= 20:
            events.append(("attacking_sequence", team, period, t0, t1))

        # 2) final-third entry: spell starts outside the third, ball enters it
        if start_x < FINAL_THIRD_X and (s.att_x >= FINAL_THIRD_X).any():
            t_in = s.loc[s.att_x >= FINAL_THIRD_X, "time_s"].iloc[0]
            events.append(("final_third_entry", team, period, t_in, t_in))

        # 3) transition: turnover, then 25+ m forward within 8 s
        if prev_team is not None and prev_team != team:
            quick = s[(s.time_s - t0 <= 8) & (s.att_x - start_x >= 25)]
            if len(quick):
                events.append(("transition", team, period, t0, quick.time_s.iloc[0]))
        prev_team = team
    return pd.DataFrame(events, columns=["type", "team", "period", "start_s", "end_s"])


if __name__ == "__main__":
    from db import save_events, load_events
    ev = detect_events(load_tracking())
    save_events(ev)
    print(ev.groupby(["type", "team"]).size())
    print(load_events().head(10).to_string(index=False))