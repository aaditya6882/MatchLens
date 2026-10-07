from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "matchlens.db"
CHROMA_DIR = DATA_DIR / "chroma"
INCOMING_DIR = DATA_DIR / "incoming"
NOTES_DIR = DATA_DIR / "knowledge"

FPS = 25
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0