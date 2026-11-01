from src.events.schema import Event
from src.events.store import EventStore
from src.events.evaluate import evaluate, format_report

# the correct answers (what a coach tagged by hand)
truth = [
    Event("m1", "shot", "home", 120, 123, source="ground_truth"),
    Event("m1", "attack", "away", 300, 312, source="ground_truth"),
    Event("m1", "key_passage", "home", 600, 606, source="ground_truth"),
]

# what the system suggested
store = EventStore(":memory:")
store.add_many([
    Event("m1", "shot", "home", 121, 123.5),          # correct
    Event("m1", "attack", "away", 450, 458),          # wrong time -> false alarm + missed
    Event("m1", "key_passage", "home", 601, 607),     # correct
])

print(format_report(evaluate(store.list_events("m1"), truth)))