import pandas as pd
from kloppy import metrica
from config import PITCH_LENGTH, PITCH_WIDTH


def _to_long(ds) -> pd.DataFrame:
    wide = ds.to_df()                     
    parts = []
    for col in wide.columns:
        if not col.endswith("_x"):
            continue
        base = col[:-2]                   
        if base == "ball":
            team = "ball"
        else:
            team = "home" if base.startswith("home") else "away"
        parts.append(pd.DataFrame({
            "frame": wide["frame_id"],
            "period": wide["period_id"],
            "time_s": wide["timestamp"].dt.total_seconds(),
            "team": team,
            "player_id": base,
            "x": (wide[f"{base}_x"] - 0.5) * PITCH_LENGTH,   # 0-1 -> metres, centred
            "y": (wide[f"{base}_y"] - 0.5) * PITCH_WIDTH,
        }).dropna(subset=["x", "y"]))
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["frame", "team", "player_id"]).reset_index(drop=True)


def load_tracking(match_id: int = 1) -> pd.DataFrame:
    """Metrica open sample data (match_id 1 or 2); downloaded on first use."""
    return _to_long(metrica.load_open_data(match_id=match_id, coordinates="metrica"))


def load_tracking_files(home_csv, away_csv) -> pd.DataFrame:
    """Same output, from your own Metrica-format CSV files."""
    return _to_long(metrica.load_tracking_csv(
        home_data=str(home_csv), away_data=str(away_csv), coordinates="metrica"))