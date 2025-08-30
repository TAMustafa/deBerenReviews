# BusinessReviews: Dutch Restaurant Review Analysis

This project analyzes Dutch-language restaurant reviews to generate descriptive analytics, a sentiment classifier, complaint taxonomy tagging, and business insights. It produces charts, an enriched CSV for BI tools, and a brief set of action-oriented suggestions based on recurring negative themes.

- Main script: `main.py`
- Data input: `data/deBerenReviews.csv`
- Outputs folder: `outputs/`

## What is analyzed
The script expects a CSV of reviews with (at least) the following columns:

- source: where the review came from
- stars or rating: numeric rating (1–5)
- review_text or review: review body in Dutch
- locatie or location: store/location string
- review_date or timestamp: review date

These are normalized internally to: `source`, `rating`, `review`, `location`, `timestamp`.

## Analysis pipeline (high-level)
1. Data loading and cleaning
   - Normalize columns, coerce ratings to [1..5]
   - Parse timestamps; derive `date` and `month`
   - Drop duplicate review texts (keep latest)
   - Label sentiment from rating:
     - 1–2 → negative
     - 3 → neutral
     - 4–5 → positive

2. Text preprocessing (Dutch)
   - Removes URLs, numbers, punctuation
   - Stopwords: NLTK Dutch + domain extras
   - Negation handling for words like "niet/geen" → `not_<token>`
   - Lemmatization with spaCy `nl_core_news_sm` if installed; otherwise NLTK Snowball stemming fallback

3. Vectorization (features)
   - Combined TF‑IDF:
     - Word n‑grams (1–2), max_features=40000
     - Character n‑grams (3–5)
   - Features are horizontally stacked (`scipy.sparse.hstack`)

4. Supervised sentiment model (3 classes)
   - Train/test split (80/20 stratified)
   - Model selection by macro F1 via small StratifiedKFold CV among:
     - LogisticRegression (several C values)
     - LinearSVC (several C values)
   - Best model is fitted and evaluated; top indicative terms are exported for linear models

5. Complaint taxonomy tagging (regex-based)
   - Categories include: service, wait_time, food_quality, portion_temp, pricing_value, ambience, order_accuracy, cleanliness
   - Per-review tags and aggregate counts by sentiment are produced

6. Outputs for BI and business insights
   - Charts for distribution and trend
   - Enriched CSV with cleaned text, sentiment, and complaint categories
   - Simple improvement suggestions based on negative complaint frequencies

## Outputs produced in `outputs/`
- ratings_distribution.png — Histogram of rating counts
- sentiment_distribution.png — Distribution of negative/neutral/positive labels
- avg_rating_over_time.png — Monthly average rating trend
- reviews_enriched.csv — Per-review dataset with:
  - source, rating, sentiment, location, month
  - review, cleaned_review
  - complaint_categories (semicolon separated)
- complaint_agg_by_sentiment.csv — Aggregated complaint category counts split by sentiment
- top_terms_per_sentiment.txt — Top discriminative terms per class (if model exposes coefficients)
- business_suggestions.txt — Short list of improvement suggestions from complaint patterns

Notes:
- The project intentionally removed word cloud generation; there is no word cloud output.
- Topic modeling and dominant topic assignment were removed to keep outputs lean and dashboard-friendly.

## Dependencies
Declared in `pyproject.toml` (Python ≥ 3.12):

- pandas ≥ 2.3.2 — data manipulation
- numpy ≥ 2.0 — numerics
- scipy ≥ 1.13 — sparse matrix utilities
- scikit-learn ≥ 1.5 — TF‑IDF, models, metrics, CV
- matplotlib ≥ 3.8 — plotting (charts saved to files)
- seaborn ≥ 0.13 — plotting aesthetics
- nltk ≥ 3.9 — Dutch stopwords and stemming fallback

Optional (for better lemmatization; code will fallback if absent):
- spaCy with `nl_core_news_sm`

## Installation
Using pip (within a virtual environment recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
# Optional: spaCy Dutch model
pip install spacy
python -m spacy download nl_core_news_sm
```

Using uv (fast, if you use uv):

```bash
# Install dependencies
uv pip install -e .
# Optional spaCy Dutch model
uv pip install spacy
python -m spacy download nl_core_news_sm
```

NLTK data: the script auto-downloads the Dutch stopwords if missing.

## Running the analysis
Run via the module entry point:

```bash
uv run python -m app.main
```

On completion, check the `outputs/` directory for generated artifacts. The script prints the chosen model, accuracy, and a classification report.

Note: Paths are resolved from the repository root, so you can run from any working directory.

## Data expectations
- File: `data/deBerenReviews.csv`
- UTF‑8 CSV with headers (case-insensitive recognized names):
  - source, stars/rating, review_text/review, locatie/location, review_date/timestamp
- Non-critical missing columns are created as empty; critical missing values in `review` or `rating` are dropped.

## Project structure
- `app/` — Modular pipeline
  - `config.py` — paths and constants
  - `io_utils.py` — I/O helpers (`load_data`, `ensure_output_dir`, `plot_and_save`)
  - `preprocess.py` — cleaning config, preprocessing utilities
  - `features.py` — TF‑IDF vectorization (`CombinedTfidf`, `vectorize_text`)
  - `modeling.py` — model selection and training
  - `complaints.py` — taxonomy and tagging
  - `eda.py` — charts generation
  - `export.py` — CSV export (lean schema)
  - `main.py` — orchestrates the full pipeline
- `data/` — Input CSVs
- `outputs/` — Generated charts, CSVs, and suggestions
- `pyproject.toml` — Dependencies and Python version
- `uv.lock` — Lockfile (if using `uv`)

## Notes and caveats
- Class imbalance is handled via `class_weight="balanced"` in models and macro‑F1 selection
- Topic modeling and word clouds have been removed from the pipeline
- The complaint taxonomy is regex/keyword-based and may be refined for your domain
- Add data quality checks or de-duplication rules as needed for your pipelines

## Exported CSV schema (lean)
File: `outputs/reviews_enriched.csv`

Columns:
- `source`
- `rating`
- `sentiment` (derived from rating: negative/neutral/positive)
- `location`
- `month` (YYYY-MM)
- `review`
- `cleaned_review`
- `complaint_categories` (semicolon-separated)
