from mplsoccer import Pitch
from geometry import attack_x


def _pitch(**kw):
    return Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, **kw)


def plot_frame(df, frame: int, title: str = ""):
    f = df[df.frame == frame]
    pitch = _pitch(line_color="grey", pitch_color="#2e7d32", line_zorder=2)
    fig, ax = pitch.draw(figsize=(9, 6))
    # mplsoccer's pitch runs 0..105 x 0..68, our data is centred on (0, 0): shift by half
    for team, color in [("home", "#d32f2f"), ("away", "#1565c0")]:
        t = f[f.team == team]
        pitch.scatter(t.x + 52.5, t.y + 34, ax=ax, c=color, s=120, edgecolors="white", zorder=3)
    b = f[f.team == "ball"]
    pitch.scatter(b.x + 52.5, b.y + 34, ax=ax, c="white", s=60, edgecolors="black", zorder=4)
    ax.set_title(title or f"frame {frame}")
    return fig


def player_heatmap(df, player_id: str):
    p = df[df.player_id == player_id]
    x = attack_x(p)                         # show every player attacking left -> right
    y = p.y.where(p.team != "away", -p.y)
    pitch = _pitch(line_color="white", pitch_color="#2e7d32")
    fig, ax = pitch.draw(figsize=(9, 6))
    stat = pitch.bin_statistic(x + 52.5, y + 34, statistic="count", bins=(21, 14))
    pitch.heatmap(stat, ax=ax, cmap="hot", alpha=0.7)
    pitch.scatter(x.mean() + 52.5, y.mean() + 34, ax=ax, c="cyan", s=200, edgecolors="black", zorder=5)
    ax.set_title(f"{player_id} heatmap (attacking left to right)")
    return fig