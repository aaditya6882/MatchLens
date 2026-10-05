import pandas as pd


def normalize_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Flip coordinates so every team attacks towards +x in every period.
    The goalkeeper is the deepest player, so his average x shows which goal
    the team defends."""
    df = df.copy()
    outf = df[df.team != "ball"]
    flip_keys = []
    for (team, period), g in outf.groupby(["team", "period"]):
        gk_x = g.groupby("player_id").x.mean().pipe(lambda s: s.loc[s.abs().idxmax()])
        if gk_x > 0:                       # defends the right goal -> attacks left
            flip_keys.append((team, period))
    for team, period in flip_keys:
        m = (df.team == team) & (df.period == period)
        df.loc[m, ["x", "y"]] *= -1
    # ball: flip with the home team's direction for that period
    for period in df.period.unique():
        if ("home", period) in flip_keys:
            m = (df.team == "ball") & (df.period == period)
            df.loc[m, ["x", "y"]] *= -1
    return df


def team_shape(df: pd.DataFrame, team: str, every: int = 25) -> pd.DataFrame:
    """Width, depth and defensive line height, one row per `every` frames (25 = 1/sec).
    Needs a direction-normalised df. Goalkeeper excluded (deepest player each frame)."""
    t = df[(df.team == team) & (df.frame % every == 0)]
    rows = []
    for (frame, period, time_s), g in t.groupby(["frame", "period", "time_s"]):
        g = g.sort_values("x")
        outfield = g.iloc[1:]              # drop deepest player (the keeper)
        if len(outfield) < 8:
            continue
        rows.append({
            "frame": frame, "period": period, "time_s": time_s,
            "width_m": outfield.y.max() - outfield.y.min(),
            "depth_m": outfield.x.max() - outfield.x.min(),
            "def_line_x": outfield.x.iloc[:4].mean(),   # 4 deepest outfield players
        })
    return pd.DataFrame(rows)
def normalize_pitch(df: pd.DataFrame) -> pd.DataFrame:
    """Flip whole periods (players AND ball together) so HOME always attacks +x
    and AWAY attacks -x. Keeps ball and players in one shared coordinate frame."""
    df = df.copy()
    for period, g in df[df.team == "home"].groupby("period"):
        gk_x = g.groupby("player_id").x.mean().pipe(lambda s: s.loc[s.abs().idxmax()])
        if gk_x > 0:
            m = df.period == period
            df.loc[m, ["x", "y"]] *= -1
    return df