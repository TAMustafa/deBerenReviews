import os
from typing import Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .config import OUTPUT_DIR


def ensure_output_dir(path: str = OUTPUT_DIR) -> None:
    os.makedirs(path, exist_ok=True)


def plot_and_save(fig, name: str) -> None:
    ensure_output_dir()
    out = os.path.join(OUTPUT_DIR, name)
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and map columns to internal schema: source, rating, review, location, timestamp"""
    df = pd.read_csv(path, header=0, encoding="utf-8", engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {
        "source": "source",
        "stars": "rating",
        "review_text": "review",
        "locatie": "location",
        "review_date": "timestamp",
        # alternates
        "rating": "rating",
        "review": "review",
        "location": "location",
        "timestamp": "timestamp",
        "date": "timestamp",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    expected = ["source", "rating", "review", "location", "timestamp"]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    df = df[expected + [c for c in df.columns if c not in expected]]
    return df
