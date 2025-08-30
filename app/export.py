import os
import pandas as pd

from .config import OUTPUT_DIR
from .io_utils import ensure_output_dir
from .complaints import complaint_taxonomy


def export_enriched_csv(df: pd.DataFrame, cleaned_texts, per_text_complaints):
    """Export review-level enriched CSV for BI tools (lean schema)."""
    ensure_output_dir()
    enriched = df.copy()
    enriched["cleaned_review"] = cleaned_texts
    # Expand complaint categories into individual columns (1/0 per category)
    categories = list(complaint_taxonomy().keys())
    for cat in categories:
        enriched[cat] = [1 if cat in cats else 0 for cats in per_text_complaints]

    cols = [
        "source", "rating", "location", "month",
        "review", "cleaned_review",
    ]
    enriched_out = enriched[cols + categories]
    enriched_out.to_csv(os.path.join(OUTPUT_DIR, "reviews_enriched.csv"), index=False)

    # Also export per-category total counts
    counts = {"category": [], "count": []}
    for cat in categories:
        counts["category"].append(cat)
        counts["count"].append(int(enriched_out[cat].sum()))
    pd.DataFrame(counts).to_csv(
        os.path.join(OUTPUT_DIR, "complaint_category_counts.csv"), index=False
    )
