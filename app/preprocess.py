import re
import string
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

# NLTK / spaCy
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None


def ensure_nltk_resources() -> None:
    """Download required NLTK resources if not present."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:  # pragma: no cover
        nltk.download("stopwords")


def try_load_spacy_nl():
    """Try to load spaCy Dutch model; return nlp or None if unavailable."""
    if spacy is None:
        return None
    try:
        return spacy.load("nl_core_news_sm")
    except Exception:  # pragma: no cover
        return None


@dataclass
class CleanConfig:
    min_rating: int = 1
    max_rating: int = 5
    neutral_rating: int = 3


def basic_clean(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    df = df.copy()
    # Strip whitespace and unify types
    for c in ["source", "review", "location"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Coerce rating to numeric and clamp to valid range
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["review", "rating"])  # must have text and rating
    df = df[(df["rating"] >= cfg.min_rating) & (df["rating"] <= cfg.max_rating)]

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    # Derive time features (strip timezone before period to avoid warnings)
    ts_naive = df["timestamp"].dt.tz_convert(None)
    df["date"] = ts_naive.dt.date
    df["month"] = ts_naive.dt.to_period("M").astype(str)

    # Drop exact duplicate reviews (keep latest)
    df = df.sort_values("timestamp").drop_duplicates(subset=["review"], keep="last")

    # Label sentiment from rating
    def label_sentiment(r: float) -> str:
        if r <= 2:
            return "negative"
        if r >= 4:
            return "positive"
        return "neutral"

    df["sentiment"] = df["rating"].apply(label_sentiment)
    return df


def preprocess_texts(texts: pd.Series) -> Tuple[List[str], List[str]]:
    """Dutch preprocessing with optional spaCy lemmatization and negation handling.

    Fallbacks to NLTK stemming if spaCy model isn't available.
    """
    ensure_nltk_resources()
    base_stop = set(stopwords.words("dutch"))
    domain_extra = {"beren", "restaurant", "eten", "drinken", "menukaart", "besteld", "bestellen", "gerechten"}
    stop_set = base_stop.union(domain_extra)
    url_re = re.compile(r"https?://\S+|www\.\S+")
    num_re = re.compile(r"\d+")
    punct_tbl = str.maketrans("", "", string.punctuation)

    nlp = try_load_spacy_nl()
    if nlp is not None:
        def clean_spacy(t: str) -> str:
            t = str(t).lower()
            t = url_re.sub(" ", t)
            t = num_re.sub(" ", t)
            doc = nlp(t)
            out_tokens: List[str] = []
            i = 0
            while i < len(doc):
                tok = doc[i]
                if tok.is_space or tok.is_punct:
                    i += 1
                    continue
                lemma = tok.lemma_.strip()
                if not lemma:
                    i += 1
                    continue
                # Negation handling: niet/geen -> not_ next token lemma
                if lemma in {"niet", "geen"} and i + 1 < len(doc):
                    nxt = doc[i + 1]
                    nxt_lemma = nxt.lemma_.strip()
                    if nxt_lemma and nxt_lemma not in stop_set and len(nxt_lemma) > 2:
                        out_tokens.append(f"not_{nxt_lemma}")
                    i += 2
                    continue
                if lemma not in stop_set and len(lemma) > 2 and lemma.isalpha():
                    out_tokens.append(lemma)
                i += 1
            return " ".join(out_tokens)

        cleaned = [clean_spacy(t) for t in texts.fillna("")]
    else:
        stemmer = SnowballStemmer("dutch")
        def clean_nltk(t: str) -> str:
            t = str(t).lower()
            t = url_re.sub(" ", t)
            t = num_re.sub(" ", t)
            t = t.translate(punct_tbl)
            tokens = []
            parts = t.split()
            i = 0
            while i < len(parts):
                w = parts[i]
                if w in {"niet", "geen"} and i + 1 < len(parts):
                    nxt = parts[i + 1]
                    if nxt not in stop_set and len(nxt) > 2:
                        tokens.append(f"not_{nxt}")
                    i += 2
                    continue
                if w not in stop_set and len(w) > 2:
                    tokens.append(stemmer.stem(w))
                i += 1
            return " ".join(tokens)
        cleaned = [clean_nltk(t) for t in texts.fillna("")]

    tokens_example = cleaned[:5]
    return cleaned, tokens_example
