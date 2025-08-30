# Central configuration
from pathlib import Path

# Resolve project root (parent of this file's directory)
_APP_DIR = Path(__file__).resolve().parent
_ROOT = _APP_DIR.parent

DATA_PATH = _ROOT / "data" / "deBerenReviews.csv"
OUTPUT_DIR = _ROOT / "outputs"
