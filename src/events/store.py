import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

from src.config import DB_PATH
from src.events.schema import Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id    TEXT NOT NULL,
    type        TEXT NOT NULL,
    team        TEXT NOT NULL,
    start_s     REAL NOT NULL,
    end_s       REAL NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    source      TEXT NOT NULL DEFAULT 'rules',
    status      TEXT NOT NULL DEFAULT 'suggested',
    note        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_events_match ON events(match_id, start_s);

CREATE TABLE IF NOT EXISTS corrections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   INTEGER NOT NULL,
    action     TEXT NOT NULL,          -- confirm | reject | edit | add
    old_json   TEXT,
    new_json   TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (event_id) REFERENCES events(id)
);
"""

COLUMNS = ("match_id", "type", "team", "start_s", "end_s",
           "confidence", "source", "status", "note")


class EventStore:
    def __init__(self, path: Path | str = DB_PATH):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._keep = None
        if self.path == ":memory:":          # keep one connection alive for tests
            self._keep = sqlite3.connect(":memory:")
            self._keep.row_factory = sqlite3.Row
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        c = self._keep or sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            if self._keep is None:
                c.close()

    @staticmethod
    def _row_to_event(r) -> Event:
        return Event(**{k: r[k] for k in r.keys()})

    def add(self, e: Event, log: bool = False) -> int:
        """Insert an event. Pass log=True when a coach adds a missed event."""
        with self._conn() as c:
            cur = c.execute(
                f"INSERT INTO events ({','.join(COLUMNS)}) "
                f"VALUES ({','.join('?' * len(COLUMNS))})",
                [getattr(e, k) for k in COLUMNS])
            e.id = cur.lastrowid
            if log:
                c.execute(
                    "INSERT INTO corrections (event_id, action, new_json) VALUES (?,?,?)",
                    (e.id, "add", json.dumps(e.to_dict())))
        return e.id

    def add_many(self, events: Iterable[Event]) -> list[int]:
        return [self.add(e) for e in events]

    def get(self, event_id: int) -> Optional[Event]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return self._row_to_event(r) if r else None

    def list_events(self, match_id: str, type: Optional[str] = None,
                    team: Optional[str] = None,
                    include_rejected: bool = False) -> list[Event]:
        q, args = "SELECT * FROM events WHERE match_id=?", [match_id]
        if type:
            q += " AND type=?"; args.append(type)
        if team:
            q += " AND team=?"; args.append(team)
        if not include_rejected:
            q += " AND status!='rejected'"
        q += " ORDER BY start_s"
        with self._conn() as c:
            return [self._row_to_event(r) for r in c.execute(q, args).fetchall()]

    # ---- coach actions: every one writes to the corrections log ----
    def confirm(self, event_id: int) -> Event:
        return self._update(event_id, "confirm", status="confirmed")

    def reject(self, event_id: int) -> Event:
        return self._update(event_id, "reject", status="rejected")

    def edit(self, event_id: int, **changes) -> Event:
        allowed = {"start_s", "end_s", "type", "team", "note"}
        bad = set(changes) - allowed
        if bad:
            raise ValueError(f"cannot edit fields: {sorted(bad)}")
        return self._update(event_id, "edit", status="edited", **changes)

    def _update(self, event_id: int, action: str, **changes) -> Event:
        with self._conn() as c:
            r = c.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
            if r is None:
                raise KeyError(f"event {event_id} not found")
            old = self._row_to_event(r)
            new = Event(**{**old.to_dict(), **changes})      # re-validates
            sets = ",".join(f"{k}=?" for k in changes)
            c.execute(f"UPDATE events SET {sets} WHERE id=?", [*changes.values(), event_id])
            c.execute(
                "INSERT INTO corrections (event_id, action, old_json, new_json) VALUES (?,?,?,?)",
                (event_id, action, json.dumps(old.to_dict()), json.dumps(new.to_dict())))
        return new

    def corrections(self, event_id: Optional[int] = None) -> list[dict]:
        q, args = "SELECT * FROM corrections", []
        if event_id is not None:
            q += " WHERE event_id=?"; args.append(event_id)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY id", args).fetchall()]