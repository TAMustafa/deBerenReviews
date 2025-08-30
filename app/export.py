import os
import pandas as pd

from .config import OUTPUT_DIR
from .io_utils import ensure_output_dir


def export_enriched_csv(df: pd.DataFrame, cleaned_texts, ml_keywords=None):
    """Export review-level enriched CSV for BI tools.

    Columns: source, rating, location, month, cleaned_review[, ml_keywords]
    """
    ensure_output_dir()
    enriched = df.copy()
    enriched["cleaned_review"] = cleaned_texts

    cols = [
        "source", "rating", "location", "month",
        "cleaned_review"
    ]
    if ml_keywords is not None:
        enriched["ml_keywords"] = ml_keywords
        cols.append("ml_keywords")
    enriched_out = enriched[cols]
    enriched_out.to_csv(os.path.join(OUTPUT_DIR, "reviews_enriched.csv"), index=False)
