import pytest
from src.events.schema import Event


def test_valid_event():
    e = Event(match_id="m1", type="shot", team="home", start_s=120.0, end_s=123.0)
    assert e.status == "suggested"
    assert e.to_dict()["type"] == "shot"


def test_rejects_unknown_type():
    with pytest.raises(ValueError):
        Event(match_id="m1", type="corner", team="home", start_s=1, end_s=2)


def test_rejects_end_before_start():
    with pytest.raises(ValueError):
        Event(match_id="m1", type="shot", team="home", start_s=5, end_s=4)


def test_rejects_bad_confidence():
    with pytest.raises(ValueError):
        Event(match_id="m1", type="shot", team="home", start_s=1, end_s=2, confidence=1.5)