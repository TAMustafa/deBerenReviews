# BusinessReviews: Dutch Restaurant Review Analysis

This project analyzes Dutch-language restaurant reviews to generate descriptive analytics, complaint taxonomy tagging, and AI-driven business insights. It produces charts, an enriched CSV for BI tools, and a brief set of action-oriented suggestions based on recurring negative themes.

- Main script: `main.py`
- Data input: `data/deBerenReviews.csv`
- Outputs folder: `outputs/`

## Analysis pipeline (high-level)
1. Data loading and cleaning
   - Normalize columns, coerce ratings to [1..5]
   - Parse timestamps; derive `date` and `month`
   - Drop duplicate review texts (keep latest)

2. Text preprocessing (Dutch)
   - Removes URLs, numbers, punctuation
   - Stopwords: NLTK Dutch + domain extras
   - Negation handling for words like "niet/geen" → `not_<token>`
   - Lemmatization with spaCy `nl_core_news_sm` if installed; otherwise NLTK Snowball stemming fallback

3. Complaint taxonomy tagging (regex-based)
   - Categories include: service, wait_time, food_quality, portion_temp, pricing_value, ambience, order_accuracy, cleanliness
   - Per-review tags and aggregate counts are produced

4. AI-driven business suggestions (optional)
   - Negative reviews are defined as rating ≤ 2
   - If enabled, an LLM (Ollama `gemma3:latest`) turns negative samples and complaint counts into actionable suggestions
   - Rule-based fallbacks are used if the LLM is disabled or fails

6. Outputs for BI and business insights
   - Charts for distribution and trend
   - Enriched CSV with cleaned text and complaint categories (one-hot columns)
   - Improvement suggestions based on negative complaint frequencies (LLM or rules)

## Outputs produced in `outputs/`
- ratings_distribution.png — Histogram of rating counts
- avg_rating_over_time.png — Monthly average rating trend
- reviews_enriched.csv — Per-review dataset with:
  - source, rating, location, month
  - review, cleaned_review
  - complaint category columns (one-hot per category)
- complaint_category_counts.csv — Aggregated complaint category totals (BigQuery-friendly)
- business_suggestions.txt — Short list of improvement suggestions (source noted: llm or rule)
- business_suggestions.csv — Same suggestions, flat schema for BigQuery with metadata

Notes:
- The project intentionally removed word cloud generation; there is no word cloud output.
- Topic modeling, sentiment classification, and top-terms exports were removed to keep outputs lean and dashboard-friendly.

## Dependencies
Declared in `pyproject.toml` (Python ≥ 3.12):

- pandas ≥ 2.3.2 — data manipulation
- matplotlib ≥ 3.8 — plotting (charts saved to files)
- seaborn ≥ 0.13 — plotting aesthetics
- nltk ≥ 3.9 — Dutch stopwords and stemming fallback
- requests ≥ 2.32 — Ollama HTTP calls (if LLM enabled)

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
Run via the module entry point (main entrypoint is `app/main.py`):

```bash
uv run python -m app.main
```

On completion, check the `outputs/` directory for generated artifacts. If LLM is enabled, the console will show LLM call status.

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
  - `complaints.py` — taxonomy and tagging
  - `eda.py` — charts generation
  - `export.py` — CSV export (lean schema)
  - `main.py` — orchestrates the full pipeline
  - `llm_suggestions.py` — calls Ollama to produce suggestions
- `data/` — Input CSVs
- `outputs/` — Generated charts, CSVs, and suggestions
- `pyproject.toml` — Dependencies and Python version
- `uv.lock` — Lockfile (if using `uv`)

## Notes and caveats
- Word clouds, topic modeling, and sentiment classification have been removed from the pipeline
- The complaint taxonomy is regex/keyword-based and may be refined for your domain
- Add data quality checks or de-duplication rules as needed for your pipelines

## Exported CSV schema (lean)
File: `outputs/reviews_enriched.csv`

Columns:
- `source`
- `rating`
- `location`
- `month` (YYYY-MM)
- `review`
- `cleaned_review`
- one-hot complaint category columns: `service`, `wait_time`, `food_quality`, `portion_temp`, `pricing_value`, `ambience`, `order_accuracy`, `cleanliness`

File: `outputs/complaint_category_counts.csv`

Columns:
- `category`
- `count`

File: `outputs/business_suggestions.csv`

Columns:
- `suggestion` (STRING)
- `source` (STRING: llm|rule)
- `model` (STRING, nullable; e.g., `ollama:gemma3:latest`)
- `generated_at` (ISO8601 UTC)

## LLM configuration (optional)
Environment variables (see `app/config.py` for defaults):

- `USE_LLM_SUGGESTIONS` (default: true)
- `OLLAMA_BASE_URL` (default: http://localhost:11434)
- `OLLAMA_MODEL` (default: gemma3:latest)
- `LLM_MAX_NEG_REVIEW_SAMPLES` (default: 100)

Make sure Ollama is running locally and the model is pulled:

```bash
ollama pull gemma3:latest
ollama serve
