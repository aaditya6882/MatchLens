from dataclasses import dataclass
from typing import Sequence

from src.events.schema import Event, EVENT_TYPES


@dataclass
class Score:
    tp: int   # predicted and real (correct)
    fp: int   # predicted but not real (false alarm)
    fn: int   # real but missed

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def match_events(pred: Sequence[Event], truth: Sequence[Event], tol_s: float = 2.0) -> Score:
    """One-to-one matching: same type and team, start times within tol_s seconds.
    Each prediction takes the closest unmatched real event."""
    unmatched = sorted(truth, key=lambda e: e.start_s)
    tp = 0
    for p in sorted(pred, key=lambda e: e.start_s):
        best, best_d = None, None
        for t in unmatched:
            if t.type != p.type or t.team != p.team:
                continue
            d = abs(t.start_s - p.start_s)
            if d <= tol_s and (best_d is None or d < best_d):
                best, best_d = t, d
        if best is not None:
            unmatched.remove(best)
            tp += 1
    return Score(tp=tp, fp=len(pred) - tp, fn=len(unmatched))


def evaluate(pred: Sequence[Event], truth: Sequence[Event], tol_s: float = 2.0) -> dict:
    """Scores per event type, plus an overall score."""
    out = {}
    for t in EVENT_TYPES:
        out[t] = match_events([e for e in pred if e.type == t],
                              [e for e in truth if e.type == t], tol_s)
    out["overall"] = match_events(pred, truth, tol_s)
    return out


def format_report(scores: dict) -> str:
    lines = [f"{'type':<12}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>8}{'recall':>8}{'F1':>8}"]
    for name, s in scores.items():
        lines.append(f"{name:<12}{s.tp:>4}{s.fp:>4}{s.fn:>4}"
                     f"{s.precision:>8.2f}{s.recall:>8.2f}{s.f1:>8.2f}")
    return "\n".join(lines)