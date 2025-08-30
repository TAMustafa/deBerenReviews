import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Support both "python -m app.main" (package) and direct script execution
try:
    from .config import DATA_PATH, OUTPUT_DIR
    from .io_utils import load_data, ensure_output_dir
    from .preprocess import CleanConfig, basic_clean, preprocess_texts
    from .eda import run_eda
    from .features import vectorize_text
    from .modeling import train_sentiment_model
    from .complaints import tag_complaints
    from .export import export_enriched_csv
except ImportError:  # running as a script without package context
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from app.config import DATA_PATH, OUTPUT_DIR
    from app.io_utils import load_data, ensure_output_dir
    from app.preprocess import CleanConfig, basic_clean, preprocess_texts
    from app.eda import run_eda
    from app.features import vectorize_text
    from app.modeling import train_sentiment_model
    from app.complaints import tag_complaints
    from app.export import export_enriched_csv


def main():
    print("Loading data from:", DATA_PATH)
    df = load_data(DATA_PATH)

    cfg = CleanConfig()
    df = basic_clean(df, cfg)

    # EDA and plots
    run_eda(df)

    # NLP preprocessing
    cleaned_texts, sample_tokens = preprocess_texts(df["review"])

    # Supervised sentiment from ratings (3 classes)
    y = df["sentiment"].values
    from sklearn.model_selection import train_test_split
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        cleaned_texts, y, test_size=0.2, random_state=42, stratify=y
    )
    vec, X_train, X_test = vectorize_text(X_train_texts, X_test_texts)

    model, model_name = train_sentiment_model(X_train, y_train)
    print(f"Chosen model: {model_name}")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Sentiment model accuracy: {acc:.3f}")

    # Identify top indicative terms per class for insights
    if hasattr(model, "coef_"):
        feature_names = np.array(vec.get_feature_names_out())
        classes = model.classes_
        top_k = 15
        insights = []
        for idx, cls in enumerate(classes):
            coefs = model.coef_[idx]
            top_idx = np.argsort(coefs)[-top_k:][::-1]
            terms = feature_names[top_idx]
            insights.append((cls, terms.tolist()))
        ensure_output_dir()
        with open(os.path.join(OUTPUT_DIR, "top_terms_per_sentiment.txt"), "w", encoding="utf-8") as f:
            for cls, terms in insights:
                f.write(f"Class: {cls}\n")
                f.write(", ".join(terms) + "\n\n")
        # Also export top terms per sentiment to CSV for BI tools
        terms_rows = []
        for cls, terms in insights:
            for rank, term in enumerate(terms, start=1):
                terms_rows.append({"sentiment": cls, "rank": rank, "term": term})
        if terms_rows:
            pd.DataFrame(terms_rows).to_csv(
                os.path.join(OUTPUT_DIR, "top_terms_per_sentiment.csv"), index=False
            )

    # Complaint tagging (all reviews + negative-only summary)
    neg_mask = df["sentiment"] == "negative"
    per_text_complaints, complaint_counts = tag_complaints(cleaned_texts)

    # Save negative-only complaint summary (counts used for suggestions)
    from collections import Counter
    neg_complaints = Counter()
    for cats, is_neg in zip(per_text_complaints, neg_mask):
        if is_neg:
            neg_complaints.update(cats)

    # Export enriched CSVs and aggregations for BI tools (slim schema)
    export_enriched_csv(df, cleaned_texts, per_text_complaints)

    # Lightweight improvement suggestions (based on negative complaint categories)
    suggestions = []
    if neg_complaints:
        if neg_complaints.get("wait_time", 0) > 0 or neg_complaints.get("service", 0) > 0:
            suggestions.append("Reduce wait times and improve service flow: adjust peak staffing and set service time KPIs.")
        if neg_complaints.get("portion_temp", 0) > 0 or neg_complaints.get("food_quality", 0) > 0:
            suggestions.append("Food quality control: enforce pass checks to avoid cold/forgotten dishes and standardize recipes.")
        if neg_complaints.get("pricing_value", 0) > 0:
            suggestions.append("Pricing perception: introduce value bundles and clarify pricing on menus.")
        if neg_complaints.get("ambience", 0) > 0:
            suggestions.append("Ambience: adjust music volume policy and review climate control standards.")
        if neg_complaints.get("cleanliness", 0) > 0:
            suggestions.append("Cleanliness: increase FOH/BOH cleaning cadence and visible hygiene checks.")
        if neg_complaints.get("order_accuracy", 0) > 0:
            suggestions.append("Order accuracy: implement order confirmation steps and expo verification.")

    ensure_output_dir()
    with open(os.path.join(OUTPUT_DIR, "business_suggestions.txt"), "w", encoding="utf-8") as f:
        if suggestions:
            f.write("Key improvement suggestions (data-driven):\n")
            for s in suggestions:
                f.write(f"- {s}\n")
        else:
            f.write("No strong recurring pain points detected in negative topics. Continue monitoring.")

    # Also export business suggestions to CSV for BI tools (always create file for stable schema)
    sugg_df = pd.DataFrame({"suggestion": suggestions})
    sugg_df.to_csv(os.path.join(OUTPUT_DIR, "business_suggestions.csv"), index=False)

    print(f"Artifacts saved to '{OUTPUT_DIR}/': charts and suggestions.")


if __name__ == "__main__":
    main()
