from dataclasses import dataclass, asdict
from typing import Optional

EVENT_TYPES = ("shot", "attack", "key_passage")
TEAMS = ("home", "away")
STATUSES = ("suggested", "confirmed", "edited", "rejected", "added")
SOURCES = ("rules", "video_model", "coach", "ground_truth")


@dataclass
class Event:
    match_id: str
    type: str
    team: str
    start_s: float
    end_s: float
    confidence: float = 1.0
    source: str = "rules"
    status: str = "suggested"
    note: str = ""
    id: Optional[int] = None

    def __post_init__(self):
        if self.type not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {self.type!r}")
        if self.team not in TEAMS:
            raise ValueError(f"unknown team: {self.team!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status: {self.status!r}")
        if self.source not in SOURCES:
            raise ValueError(f"unknown source: {self.source!r}")
        if self.end_s < self.start_s:
            raise ValueError("end_s must be >= start_s")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)