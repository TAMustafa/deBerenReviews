# Central configuration
import os
from pathlib import Path

# Resolve project root (parent of this file's directory)
_APP_DIR = Path(__file__).resolve().parent
_ROOT = _APP_DIR.parent

DATA_PATH = _ROOT / "data" / "deBerenReviews.csv"
OUTPUT_DIR = _ROOT / "outputs"

# LLM / Ollama configuration
USE_LLM_SUGGESTIONS: bool = os.getenv("USE_LLM_SUGGESTIONS", "true").lower() in {"1", "true", "yes"}
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Prefer a concrete tag or default family without :latest for reproducibility
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:latest")
LLM_MAX_NEG_REVIEW_SAMPLES: int = int(os.getenv("LLM_MAX_NEG_REVIEW_SAMPLES", "100"))

# Sentiment (Nederlands) configuratie – RobBERT v2 is standaard
HF_DUTCH_SENTIMENT_MODEL: str = os.getenv(
    "HF_DUTCH_SENTIMENT_MODEL",
    "DTAI-KULeuven/robbert-v2-dutch-sentiment",
)
