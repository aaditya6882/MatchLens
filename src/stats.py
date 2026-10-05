import numpy as np
import pandas as pd
from loader import load_tracking

FPS = 25
MAX_SPEED = 12.0      # m/s, faster than this is a tracking glitch
SPRINT_SPEED = 7.0    # m/s (about 25 km/h)
SPRINT_MIN_S = 1.0


def _runs(mask: np.ndarray):
    """Lengths of consecutive True runs."""
    padded = np.concatenate(([0], mask.astype(int), [0]))
    d = np.diff(padded)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return ends - starts


def player_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for pid, g in df[df.team != "ball"].groupby("player_id"):
        dist, speeds, sprints = 0.0, [], 0
        for _, p in g.groupby("period"):
            p = p.sort_values("frame")
            dt = p.time_s.diff().to_numpy()
            step = np.hypot(p.x.diff(), p.y.diff()).to_numpy()
            ok = (dt > 0) & (dt < 0.1)             # skip gaps in the data
            speed = np.where(ok, step / np.where(dt > 0, dt, np.nan), np.nan)
            speed[speed > MAX_SPEED] = np.nan      # drop glitches
            smooth = pd.Series(speed).rolling(12, min_periods=6, center=True).mean().to_numpy()
            dist += np.nansum(np.where(ok & ~np.isnan(speed), step, 0))
            speeds.append(smooth)
            fast = np.nan_to_num(smooth) > SPRINT_SPEED
            sprints += int((_runs(fast) >= SPRINT_MIN_S * FPS).sum())
        allspeed = np.concatenate(speeds)
        out.append({
            "player_id": pid,
            "team": g.team.iloc[0],
            "minutes": round(len(g) / FPS / 60, 1),
            "distance_km": round(dist / 1000, 2),
            "top_speed_kmh": round(np.nanmax(allspeed) * 3.6, 1),
            "sprints": sprints,
        })
    return pd.DataFrame(out).sort_values(["team", "player_id"]).reset_index(drop=True)


if __name__ == "__main__":
    df = load_tracking()
    print(player_stats(df).to_string(index=False))