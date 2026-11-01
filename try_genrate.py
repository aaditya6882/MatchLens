"""Gate validation script:
Generates 5 distinct synthetic matches, verifies their tracking tables and truth.csv files,
and plots a pitch frame from each.
"""
from pathlib import Path
import pandas as pd

from src.config import DATA_DIR
from src.simulate.generate import generate_match, plot_frame

MATCH_COUNT = 5
DURATION_S = 180.0  # 3 minutes per match at 25 fps = 4,500 frames = 103,500 rows per match

print(f"Generating {MATCH_COUNT} synthetic matches ({DURATION_S}s each)...")
matches_dir = DATA_DIR / "matches"

for i in range(1, MATCH_COUNT + 1):
    match_id = f"sim_match_{i}"
    seed = 100 + i
    print(f"\n--- Generating {match_id} (seed={seed}) ---")
    data = generate_match(match_id=match_id, duration_s=DURATION_S, seed=seed)

    match_dir = matches_dir / match_id
    track_file = match_dir / "tracking.csv"
    truth_file = match_dir / "truth.csv"

    assert track_file.exists(), f"Missing tracking file for {match_id}"
    assert truth_file.exists(), f"Missing truth file for {match_id}"

    df_track = pd.read_csv(track_file)
    df_truth = pd.read_csv(truth_file)

    print(f"  Tracking rows: {len(df_track):,} | Columns: {list(df_track.columns)}")
    print(f"  Truth events:  {len(df_truth)} events logged")
    for _, row in df_truth.iterrows():
        print(f"    - {row['type']:12s} ({row['team']:4s}) from {row['start_s']:6.2f}s to {row['end_s']:6.2f}s | {row['note']}")

    # Save a plotted pitch frame
    frame_to_plot = 100 * i
    img_path = match_dir / f"frame_{frame_to_plot}.png"
    plot_frame(df_track, frame=frame_to_plot, title=f"{match_id} - Frame {frame_to_plot}", save_path=img_path)
    print(f"  Saved pitch plot: {img_path}")

print("\nAll 5 matches successfully generated with tracking tables, truth files, and pitch plots.")
