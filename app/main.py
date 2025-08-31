from datetime import datetime, UTC
import pandas as pd

# Main entrypoint: import directly from the app package
from app.config import DATA_PATH, OUTPUT_DIR, USE_LLM_SUGGESTIONS, LLM_MAX_NEG_REVIEW_SAMPLES, OLLAMA_MODEL
from app.io_utils import load_data, ensure_output_dir
from app.preprocess import CleanConfig, basic_clean, preprocess_texts
from app.eda import run_eda
from app.complaints import tag_complaints
from app.export import (
    export_enriched_csv,
    export_pain_points,
    export_average_rating,
    export_sentiment_aggregate,
)
from app.llm_suggestions import generate_suggestions_llm
from app.ml_keywords import extract_keywords_controlled
from app.sentiment import compute_sentiment, plot_sentiment_images


def main():
    print("Data laden van:", DATA_PATH)
    df = load_data(DATA_PATH)

    cfg = CleanConfig()
    df = basic_clean(df, cfg)

    # EDA en grafieken
    run_eda(df)

    # NLP preprocessing
    cleaned_texts, _sample_tokens = preprocess_texts(df["review"])

    # Klachttagging blijft voor analyses/suggesties; niet als kolommen geëxporteerd
    # Gebruik rating als proxy voor negatief (<= 2 sterren)
    neg_mask = df["rating"].astype(float) <= 2
    per_text_complaints, _complaint_counts = tag_complaints(cleaned_texts)

    # Save negative-only complaint summary (counts used for suggestions)
    from collections import Counter
    neg_complaints = Counter()
    for cats, is_neg in zip(per_text_complaints, neg_mask):
        if is_neg:
            neg_complaints.update(cats)

    # Exporteer meest genoemde pijnpunten (op basis van negatieve reviews)
    export_pain_points(neg_complaints)

    # Exporteer totale gemiddelde beoordeling
    avg_rating = float(pd.to_numeric(df["rating"]).mean()) if not df.empty else float('nan')
    export_average_rating(avg_rating)

    # ML-based keywords using controlled vocabulary
    ml_keywords = extract_keywords_controlled(cleaned_texts)

    # Sentimentanalyse over opgeschoonde teksten
    polarities, subjectivities, labels = compute_sentiment(cleaned_texts)
    # Sentimentafbeeldingen en totalen
    plot_sentiment_images(polarities, labels)
    export_sentiment_aggregate(labels)

    # Export enriched CSV with ML keywords and sentiment columns
    export_enriched_csv(
        df,
        cleaned_texts,
        ml_keywords=ml_keywords,
        sentiment={
            "polarity": polarities,
            "subjectivity": subjectivities,
            "label": labels,
        },
    )

    # Regelgebaseerde baseline-suggesties (beperkt)
    suggestions_rule = []

    # LLM-suggesties via Ollama
    suggestions_llm = []
    if USE_LLM_SUGGESTIONS:
        # Bouw steekproef van negatieve reviews (gebruik opgeschoonde tekst)
        neg_reviews = [t for t, is_neg in zip(cleaned_texts, neg_mask) if is_neg]
        if len(neg_reviews) > LLM_MAX_NEG_REVIEW_SAMPLES:
            neg_reviews = neg_reviews[:LLM_MAX_NEG_REVIEW_SAMPLES]
        # Converteer Counter naar dict
        complaint_counts_dict = {k: int(v) for k, v in neg_complaints.items()}
        print("[LLM] Ollama aanroepen voor bedrijfssuggesties…")
        suggestions_llm = generate_suggestions_llm(neg_reviews, complaint_counts_dict)
        print(f"[LLM] {len(suggestions_llm)} suggesties ontvangen van LLM.")

    # Merge with priority to LLM output; fallback to rules; deduplicate
    suggestions = suggestions_llm or suggestions_rule
    seen = set()
    suggestions = [s for s in suggestions if not (s in seen or seen.add(s))]
    source = "llm" if suggestions_llm else ("rule" if suggestions_rule else "none")
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    ensure_output_dir()
    with (OUTPUT_DIR / "business_suggestions.txt").open("w", encoding="utf-8") as f:
        if suggestions:
            bron = "llm" if suggestions_llm else ("regel" if suggestions_rule else "geen")
            f.write(f"Belangrijkste verbetersuggesties (bron={bron})\n")
            for s in suggestions:
                f.write(f"- {s}\n")
        else:
            f.write("Geen sterke, terugkerende pijnpunten gevonden in negatieve reviews. Blijf monitoren.")

    # Exporteer bedrijfssuggesties naar CSV
    # Schema: suggestie (STRING), bron (STRING), model (STRING, nullable), gegenereerd_op (TIMESTAMP ISO8601)
    model_name = (f"ollama:{OLLAMA_MODEL}" if source == "llm" else None)
    sugg_df = pd.DataFrame({
        "suggestie": suggestions,
        "bron": ["llm" if suggestions_llm else ("regel" if suggestions_rule else "geen")] * len(suggestions),
        "model": [model_name] * len(suggestions),
        "gegenereerd_op": [generated_at] * len(suggestions),
    })
    sugg_df.to_csv(OUTPUT_DIR / "business_suggestions.csv", index=False)

    print(f"Artefacten opgeslagen in '{OUTPUT_DIR}/': grafieken en suggesties.")


if __name__ == "__main__":
    main()
