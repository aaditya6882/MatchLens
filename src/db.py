import sqlite3
import pandas as pd

DB_PATH = "../data/matchlens.db"


def conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER, type TEXT, team TEXT, period INTEGER,
        start_s REAL, end_s REAL,
        status TEXT DEFAULT 'pending'   -- pending / accepted / rejected
    )""")
    return c


def save_events(events: pd.DataFrame, match_id: int = 1):
    c = conn()
    c.execute("DELETE FROM events WHERE match_id=? AND status='pending'", (match_id,))
    events = events.assign(match_id=match_id)[["match_id", "type", "team", "period", "start_s", "end_s"]]
    events.to_sql("events", c, if_exists="append", index=False)
    c.commit()
    c.close()


def load_events(match_id: int = 1) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql("SELECT * FROM events WHERE match_id=? ORDER BY period, start_s", c, params=(match_id,))
    c.close()
    return df