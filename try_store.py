from src.events.schema import Event
from src.events.store import EventStore

store = EventStore()   # creates data/matchlens.db
i = store.add(Event("m1", "shot", "home", 120, 123))
store.edit(i, start_s=121.0)
store.confirm(i)

print(store.list_events("m1"))
print(store.corrections(i))