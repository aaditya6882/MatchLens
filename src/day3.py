import sys
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from loader import load_tracking
from geometry import normalize_direction, team_shape

df = normalize_direction(load_tracking())

for team in ["home", "away"]:
    s = team_shape(df, team)
    print(team, s[["width_m", "depth_m", "def_line_x"]].mean().round(1).to_dict())

player = sys.argv[1] if len(sys.argv) > 1 else "home_8"
p = df[df.player_id == player]
print(player, "average position (x, y):", round(p.x.mean(), 1), round(p.y.mean(), 1))

pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, line_color="white", pitch_color="#2e7d32")
fig, ax = pitch.draw(figsize=(10, 6.5))
stat = pitch.bin_statistic(p.x + 52.5, p.y + 34, statistic="count", bins=(21, 14))
pitch.heatmap(stat, ax=ax, cmap="hot", alpha=0.7)
pitch.scatter(p.x.mean() + 52.5, p.y.mean() + 34, ax=ax, c="cyan", s=200, edgecolors="black", zorder=5)
ax.set_title(f"{player} heatmap (attacking left to right)")
fig.savefig("heatmap.png", dpi=110, bbox_inches="tight")
print("saved heatmap.png")