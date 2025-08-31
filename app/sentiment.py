from typing import List, Tuple, Optional
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .config import HF_DUTCH_SENTIMENT_MODEL
from .io_utils import ensure_output_dir, plot_and_save


def compute_sentiment(texts: List[str]) -> Tuple[List[float], Optional[List[float]], List[str]]:
    """Bepaal sentiment met RobBERT v2 (Nederlands).

    - Polariteit: gesigneerde score (positief = +score, negatief = -score, neutraal = 0).
    - Subjectiviteit: niet beschikbaar bij dit model -> None.
    - Labels: Nederlandstalig (negatief, neutraal, positief).
    """
    texts = [t if isinstance(t, str) else "" for t in texts]
    from transformers import pipeline  # type: ignore

    clf = pipeline("text-classification", model=HF_DUTCH_SENTIMENT_MODEL, top_k=None)
    results = clf(texts, truncation=True)
    polarities: List[float] = []
    labels_nl: List[str] = []

    def to_nl(lbl: str) -> str:
        l = lbl.upper()
        if "POS" in l:
            return "positief"
        if "NEG" in l:
            return "negatief"
        return "neutraal"

    for item in results:
        # item kan dict of lijst zijn; neem hoogste score
        if isinstance(item, list) and item:
            best = max(item, key=lambda x: float(x.get("score", 0.0)))
        else:
            best = item  # type: ignore
        lbl_nl = to_nl(str(best.get("label", "neutraal")))
        score = float(best.get("score", 0.0))
        if lbl_nl == "positief":
            pol = +score
        elif lbl_nl == "negatief":
            pol = -score
        else:
            pol = 0.0
        polarities.append(pol)
        labels_nl.append(lbl_nl)

    # Subjectiviteit niet ondersteund -> None
    return polarities, None, labels_nl


def plot_sentiment_images(polarities: List[float], labels: List[str]) -> None:
    """Maak sentimentafbeeldingen: histogram van polariteit en staafdiagram van labels (NL)."""
    ensure_output_dir()

    # Polarity distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(polarities, bins=30, kde=True, ax=ax, color="#4c78a8")
    ax.set_title("Verdeling van sentimentpolariteit")
    ax.set_xlabel("Polariteit (-1 tot 1)")
    plot_and_save(fig, "sentiment_polarity_distribution.png")

    # Label counts
    fig, ax = plt.subplots(figsize=(5, 4))
    # Map naar NL labels voor de visualisatie
    label_map = {"negative": "negatief", "neutral": "neutraal", "positive": "positief"}
    labels_nl = [label_map.get(str(x).lower(), str(x)) for x in labels]
    s = pd.Series(labels_nl, name="label").value_counts().reindex(["negatief", "neutraal", "positief"], fill_value=0)
    df_counts = s.rename_axis("label").reset_index(name="aantal")
    sns.barplot(data=df_counts, x="label", y="aantal", hue="label", ax=ax, palette=["#d62728", "#7f7f7f", "#2ca02c"]) 
    if ax.legend_:
        ax.legend_.remove()
    ax.set_title("Aantal reviews per sentiment")
    ax.set_xlabel("")
    ax.set_ylabel("Aantal")
    plot_and_save(fig, "sentiment_label_counts.png")
