import re
from typing import List, Tuple
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _build_vectorizer(min_df: int = 2, max_df: float = 0.9, ngram_range: Tuple[int, int] = (1, 2)) -> TfidfVectorizer:
    # Lightweight Dutch/EN stop words tailored for reviews; avoids nltk downloads
    domain_stop = [
        "eten", "restaurant", "beren", "de", "het", "een", "en", "ook", "maar", "wij", "ik", "jij",
        "hij", "zij", "ze", "hun", "bij", "met", "van", "voor", "na", "dan", "als", "die", "dat",
        "this", "that", "the", "and", "or", "but", "very", "really", "just", "was", "were", "are",
        "is", "am", "to", "of", "in", "on", "at", "it", "they", "we", "you", "i",
    ]
    # Load extra stopwords from app/stopwords.txt if exists
    try:
        here = Path(__file__).resolve().parent
        sw_path = here / "stopwords.txt"
        if sw_path.exists():
            with sw_path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    domain_stop.append(s)
    except Exception:
        pass
    token_pattern = r"(?u)\b[a-zA-Zà-ÿ_]{2,}\b"  # keep underscores from preprocessing like not_lekker
    return TfidfVectorizer(
        lowercase=True,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        token_pattern=token_pattern,
        stop_words=domain_stop,
        norm="l2",
        sublinear_tf=True,
    )


def extract_keywords_tfidf(texts: List[str], top_k: int = 5,
                            min_df: int = 2, max_df: float = 0.9,
                            ngram_range: Tuple[int, int] = (1, 2)) -> List[str]:
    """Return top_k TF-IDF keywords per text as a comma-separated string.

    - Works unsupervised over the corpus of cleaned_review.
    - Filters duplicates, very short tokens, and overlapping n-grams.
    """
    if not texts:
        return []

    vectorizer = _build_vectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())

    results: List[str] = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            results.append("")
            continue
        indices = row.indices
        data = row.data
        order = np.argsort(-data)  # descending by tf-idf weight
        picked: List[str] = []
        seen = set()

        def overlaps(term: str) -> bool:
            # Avoid adding terms fully contained within already-picked terms (and vice versa)
            for t in picked:
                if term in t or t in term:
                    return True
            return False

        for idx in order:
            term = vocab[indices[idx]]
            if len(term) < 3:
                continue
            if term in seen:
                continue
            if overlaps(term):
                continue
            seen.add(term)
            picked.append(term)
            if len(picked) >= top_k:
                break
        results.append(", ".join(picked))

    return results


def extract_keywords_controlled(texts: List[str]) -> List[str]:
    """Map each review to a small set of standardized keywords.

    Vocabulary (extendable):
    - lange_wachten
    - duur
    - service
    - airco
    - eten_koud
    - bestelling_fout
    - hygiene
    - lawaai
    """
    vocab_order = [
        "lange_wachten",
        "duur",
        "service",
        "airco",
        "eten_koud",
        "bestelling_fout",
        "hygiene",
        "lawaai",
    ]

    # Regex patterns per keyword (lowercased input expected)
    patterns = {
        # lange_wachten
        "lange_wachten": re.compile(
            r"\b(lang|lange|lang(e)?\s+moet(en)?\s+wacht(en)?|wachttijd|lang\s+duurd|lang\s+duurt|half\s+uur|kwartier)\b"
        ),
        # duur / pricing complaints
        "duur": re.compile(r"\b(duur|prij(s|z)ig|overpriced|te\s+duur|te\s+duur|euro\s+\d{2,})\b"),
        # service / bediening complaints
        "service": re.compile(
            r"\b(service|bedien(ing)?|personeel|personel|onvriendelijk|genegeerd|slecht\s+bedien)\b"
        ),
        # airco / temperature ambience
        "airco": re.compile(r"\b(airco|airconditioning|benauwd|heet\s+binnen|warm\s+binnen|geen\s+airco)\b"),
        # food temperature
        "eten_koud": re.compile(r"\b(lauw|koud(e)?\s+eten|eten\s+koud|afgekoeld|niet\s+warm)\b|not_koud|eten_koud"),
        # order accuracy
        "bestelling_fout": re.compile(r"\b(bestelling\s+fout|verkeerd(e)?\s+bestelling|mis(s)?ing|vergeten)\b|bestelling\s*fout|not_bestelling"),
        # cleanliness
        "hygiene": re.compile(r"\b(vies|smerig|vuil|hygi[eë]ne|vliegen|insecten|schimmel)\b"),
        # noise/ambience
        "lawaai": re.compile(r"\b(lawaai|hard(e)?\s+muziek|herrie|druk(te)?)\b"),
    }

    out: List[str] = []
    for t in texts:
        s = (t or "").lower()
        found = []
        for key in vocab_order:
            rx = patterns[key]
            if rx.search(s):
                found.append(key)
        out.append(", ".join(found))
    return out
