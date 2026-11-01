import pytest

from src.events.schema import Event
from src.events.store import EventStore


def ev(t=10.0, type="shot", team="home", **kw):
    return Event(match_id="m1", type=type, team=team, start_s=t, end_s=t + 2, **kw)


def test_add_and_list_sorted():
    s = EventStore(":memory:")
    s.add_many([ev(30), ev(10), ev(20)])
    assert [e.start_s for e in s.list_events("m1")] == [10, 20, 30]


def test_confirm_reject_edit_are_logged():
    s = EventStore(":memory:")
    i = s.add(ev(10))
    s.confirm(i)
    s.edit(i, start_s=11.0, end_s=13.0)
    s.reject(i)
    actions = [c["action"] for c in s.corrections(i)]
    assert actions == ["confirm", "edit", "reject"]
    assert s.list_events("m1") == []                       # rejected are hidden
    assert len(s.list_events("m1", include_rejected=True)) == 1


def test_edit_validates_and_blocks_protected_fields():
    s = EventStore(":memory:")
    i = s.add(ev(10))
    with pytest.raises(ValueError):
        s.edit(i, status="confirmed")                      # status is not editable directly
    with pytest.raises(ValueError):
        s.edit(i, start_s=99.0)                            # end_s (12) < start_s


def test_filters():
    s = EventStore(":memory:")
    s.add_many([ev(10, "shot", "home"), ev(20, "attack", "away")])
    assert len(s.list_events("m1", type="shot")) == 1
    assert len(s.list_events("m1", team="away")) == 1