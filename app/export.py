import os
import pandas as pd

from .config import OUTPUT_DIR
from .io_utils import ensure_output_dir


def export_enriched_csv(df: pd.DataFrame, cleaned_texts, per_text_complaints):
    """Export review-level enriched CSV for BI tools (lean schema)."""
    ensure_output_dir()
    enriched = df.copy()
    enriched["cleaned_review"] = cleaned_texts
    cats_str = [";".join(c) if c else "" for c in per_text_complaints]
    enriched["complaint_categories"] = cats_str

    cols = [
        "source", "rating", "sentiment", "location", "month",
        "review", "cleaned_review", "complaint_categories",
    ]
    enriched_out = enriched[cols]
    enriched_out.to_csv(os.path.join(OUTPUT_DIR, "reviews_enriched.csv"), index=False)
