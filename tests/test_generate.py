from pathlib import Path
import pandas as pd
import pytest

from src.simulate.generate import generate_match, plot_frame, FPS


def test_reproducible_with_seed(tmp_path):
    m1 = generate_match("test1", duration_s=10.0, seed=123, output_dir=tmp_path / "m1")
    m2 = generate_match("test2", duration_s=10.0, seed=123, output_dir=tmp_path / "m2")

    pd.testing.assert_frame_equal(m1.tracking_df.drop(columns=["frame"]), m2.tracking_df.drop(columns=["frame"]))
    assert len(m1.truth_events) == len(m2.truth_events)
    for e1, e2 in zip(m1.truth_events, m2.truth_events):
        assert e1.type == e2.type
        assert e1.team == e2.team
        assert e1.start_s == e2.start_s
        assert e1.end_s == e2.end_s


def test_tracking_table_columns_and_row_count(tmp_path):
    duration_s = 5.0
    m = generate_match("test_cols", duration_s=duration_s, seed=42, output_dir=tmp_path)
    expected_frames = int(duration_s * FPS)
    # 22 players + 1 ball = 23 rows per frame
    assert len(m.tracking_df) == expected_frames * 23
    assert set(m.tracking_df.columns) == {"frame", "time_s", "team", "player_id", "x", "y"}


def test_files_written_automatically(tmp_path):
    out_dir = tmp_path / "match_out"
    generate_match("test_write", duration_s=60.0, seed=99, output_dir=out_dir)

    assert (out_dir / "tracking.csv").exists()
    assert (out_dir / "truth.csv").exists()

    df_track = pd.read_csv(out_dir / "tracking.csv")
    df_truth = pd.read_csv(out_dir / "truth.csv")
    assert not df_track.empty
    assert not df_truth.empty


def test_events_logged(tmp_path):
    m = generate_match("test_events", duration_s=90.0, seed=77, output_dir=tmp_path)
    types = {e.type for e in m.truth_events}
    assert "shot" in types
    assert "attack" in types
    assert "key_passage" in types


def test_plot_frame_saves_image(tmp_path):
    m = generate_match("test_plot", duration_s=5.0, seed=10, output_dir=tmp_path)
    img_path = tmp_path / "pitch.png"
    plot_frame(m.tracking_df, frame=0, save_path=img_path)
    assert img_path.exists()
    assert img_path.stat().st_size > 0
