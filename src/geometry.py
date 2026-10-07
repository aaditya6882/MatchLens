import numpy as np
import pandas as pd


def normalize_pitch(df: pd.DataFrame) -> pd.DataFrame:
    """After this, HOME attacks towards +x and AWAY towards -x in every period,
    and the ball stays in the same coordinate frame as the players."""
    df = df.copy()
    for period, g in df[df.team == "home"].groupby("period"):
        means = g.groupby("player_id").x.mean()
        gk_x = means.loc[means.abs().idxmax()]     # the keeper sits furthest from the centre
        if gk_x > 0:                               # home defends the right goal -> flip the period
            m = df.period == period
            df.loc[m, ["x", "y"]] *= -1
    return df


def attack_x(df: pd.DataFrame) -> pd.Series:
    """Progress towards each team's OWN attacking goal (0 = halfway, +52.5 = goal line).
    Needs a normalized df."""
    return pd.Series(np.where(df.team == "away", -df.x, df.x), index=df.index)