import pandas as pd
from kloppy import metrica

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0

def load_tracking(match_id: int = 1) -> pd.DataFrame:
    ds = metrica.load_open_data(match_id=match_id, coordinates="metrica")
    wide = ds.to_df()
    rows = []
    for col in wide.columns:
        if not col.endswith("_x"):
            continue
        base = col[:-2]
        if base == "ball":
            team, pid = "ball", "ball"
        else:
            pid = base
            team = "home" if pid.startswith("home") else "away"
        sub = pd.DataFrame({
            "frame": wide["frame_id"],
            "period": wide["period_id"],
            "time_s": wide["timestamp"].dt.total_seconds(),
            "team": team,
            "player_id": pid,
            "x": (wide[f"{base}_x"] - 0.5) * PITCH_LENGTH,
            "y": (wide[f"{base}_y"] - 0.5) * PITCH_WIDTH,
        }).dropna(subset=["x", "y"])
        rows.append(sub)
    df = pd.concat(rows, ignore_index=True).sort_values(["frame", "team", "player_id"])
    return df.reset_index(drop=True)

if __name__ == "__main__":
    df = load_tracking()
    print(df.head())
    print(df.player_id.nunique(), "ids;", df.frame.nunique(), "frames")