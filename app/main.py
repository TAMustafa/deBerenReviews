import os
from datetime import datetime, UTC
import pandas as pd

# Main entrypoint: import directly from the app package
from app.config import DATA_PATH, OUTPUT_DIR, USE_LLM_SUGGESTIONS, LLM_MAX_NEG_REVIEW_SAMPLES
from app.io_utils import load_data, ensure_output_dir
from app.preprocess import CleanConfig, basic_clean, preprocess_texts
from app.eda import run_eda
from app.complaints import tag_complaints
from app.export import export_enriched_csv
from app.llm_suggestions import generate_suggestions_llm
from app.ml_keywords import extract_keywords_controlled


def main():
    print("Loading data from:", DATA_PATH)
    df = load_data(DATA_PATH)

    cfg = CleanConfig()
    df = basic_clean(df, cfg)

    # EDA and plots
    run_eda(df)

    # NLP preprocessing
    cleaned_texts, sample_tokens = preprocess_texts(df["review"])

    # Complaint tagging kept for analytics/suggestions; but not exported as columns
    # Use rating as a proxy for negatives (<= 2 stars)
    neg_mask = df["rating"].astype(float) <= 2
    per_text_complaints, complaint_counts = tag_complaints(cleaned_texts)

    # Save negative-only complaint summary (counts used for suggestions)
    from collections import Counter
    neg_complaints = Counter()
    for cats, is_neg in zip(per_text_complaints, neg_mask):
        if is_neg:
            neg_complaints.update(cats)

    # ML-based keywords using controlled vocabulary
    ml_keywords = extract_keywords_controlled(cleaned_texts)

    # Export enriched CSV with ML keywords column
    export_enriched_csv(df, cleaned_texts, ml_keywords=ml_keywords)

    # Rule-based baseline suggestions (kept minimal)
    suggestions_rule = []

    # LLM-generated suggestions via Ollama (gemma3:latest)
    suggestions_llm = []
    if USE_LLM_SUGGESTIONS:
        # Build negative reviews sample (use cleaned text for clarity)
        neg_reviews = [t for t, is_neg in zip(cleaned_texts, neg_mask) if is_neg]
        if len(neg_reviews) > LLM_MAX_NEG_REVIEW_SAMPLES:
            neg_reviews = neg_reviews[:LLM_MAX_NEG_REVIEW_SAMPLES]
        # Convert Counter to plain dict
        complaint_counts_dict = {k: int(v) for k, v in neg_complaints.items()}
        print("[LLM] Calling Ollama for business suggestions…")
        suggestions_llm = generate_suggestions_llm(neg_reviews, complaint_counts_dict)
        print(f"[LLM] Received {len(suggestions_llm)} suggestions from LLM.")

    # Merge with priority to LLM output; fallback to rules; deduplicate
    suggestions = suggestions_llm or suggestions_rule
    seen = set()
    suggestions = [s for s in suggestions if not (s in seen or seen.add(s))]
    source = "llm" if suggestions_llm else ("rule" if suggestions_rule else "none")
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    ensure_output_dir()
    with open(os.path.join(OUTPUT_DIR, "business_suggestions.txt"), "w", encoding="utf-8") as f:
        if suggestions:
            f.write(f"Key improvement suggestions (source={source})\n")
            for s in suggestions:
                f.write(f"- {s}\n")
        else:
            f.write("No strong recurring pain points detected in negative topics. Continue monitoring.")

    # Export business suggestions to CSV for BigQuery ingestion
    # Schema: suggestion (STRING), source (STRING), model (STRING, nullable), generated_at (TIMESTAMP as ISO8601)
    model_name = "ollama:gemma3:latest" if source == "llm" else None
    sugg_df = pd.DataFrame({
        "suggestion": suggestions,
        "source": [source] * len(suggestions),
        "model": [model_name] * len(suggestions),
        "generated_at": [generated_at] * len(suggestions),
    })
    sugg_df.to_csv(os.path.join(OUTPUT_DIR, "business_suggestions.csv"), index=False)

    print(f"Artifacts saved to '{OUTPUT_DIR}/': charts and suggestions.")


if __name__ == "__main__":
    main()
