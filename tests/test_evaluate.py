from src.events.schema import Event
from src.events.evaluate import match_events, evaluate


def ev(t=10.0, type="shot", team="home", **kw):
    return Event(match_id="m1", type=type, team=team, start_s=t, end_s=t + 2, **kw)


def test_perfect_match():
    sc = match_events([ev(10), ev(50)], [ev(10.5), ev(49)])
    assert (sc.tp, sc.fp, sc.fn) == (2, 0, 0)
    assert sc.f1 == 1.0


def test_outside_tolerance_is_fp_and_fn():
    sc = match_events([ev(10)], [ev(20)], tol_s=2.0)
    assert (sc.tp, sc.fp, sc.fn) == (0, 1, 1)


def test_one_truth_cannot_match_two_predictions():
    sc = match_events([ev(10), ev(10.5)], [ev(10)])
    assert (sc.tp, sc.fp, sc.fn) == (1, 1, 0)


def test_type_and_team_must_match():
    assert match_events([ev(10, "shot", "home")], [ev(10, "shot", "away")]).tp == 0
    assert match_events([ev(10, "shot")], [ev(10, "attack")]).tp == 0


def test_per_type_report():
    r = evaluate([ev(10, "shot"), ev(40, "attack")], [ev(10, "shot")])
    assert r["shot"].recall == 1.0
    assert r["attack"].precision == 0.0
    assert r["overall"].fp == 1