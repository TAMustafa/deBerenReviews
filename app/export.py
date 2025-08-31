import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .config import OUTPUT_DIR
from .io_utils import ensure_output_dir, plot_and_save


def export_enriched_csv(df: pd.DataFrame, cleaned_texts, ml_keywords=None, sentiment=None):
    """Exporteer review-niveau CSV/XLSX met verrijkte gegevens (Nederlands).

    Kolommen: bron, beoordeling, locatie, maand, opgeschoonde_review[, ml_trefwoorden][, polariteit, subjectiviteit, sentiment]
    """
    ensure_output_dir()
    enriched = df.copy()
    enriched["opgeschoonde_review"] = cleaned_texts

    # Verwachte invoerkolommen blijven ongewijzigd in df, maar uitvoerkolommen worden NL
    # Map basisvelden naar NL namen
    out = pd.DataFrame({
        "bron": enriched.get("source"),
        "beoordeling": enriched.get("rating"),
        "locatie": enriched.get("location"),
        "maand": enriched.get("month"),
        "opgeschoonde_review": enriched.get("opgeschoonde_review"),
    })
    if ml_keywords is not None:
        out["ml_trefwoorden"] = ml_keywords
    if sentiment is not None:
        pol = sentiment.get("polarity")
        sub = sentiment.get("subjectivity")
        lab = sentiment.get("label")
        if pol is not None:
            out["polariteit"] = pol
        if sub is not None:
            out["subjectiviteit"] = sub
        if lab is not None:
            # Map Engelstalige labels naar Nederlands
            label_map = {"negative": "negatief", "neutral": "neutraal", "positive": "positief"}
            out["sentiment"] = [label_map.get(str(x).lower(), str(x)) for x in lab]
    out.to_excel(OUTPUT_DIR / "reviews_enriched.xlsx", index=False)
    out.to_csv(OUTPUT_DIR / "reviews_enriched.csv", index=False)


def export_pain_points(counts: dict, top_n: int = 10) -> None:
    """Exporteer meest genoemde pijnpunten als CSV en staafdiagram (Nederlands)."""
    ensure_output_dir()
    # Map interne categorie-sleutels naar Nederlandse labels
    nl_map = {
        "service": "service",
        "wait_time": "wachttijd",
        "food_quality": "voedselkwaliteit",
        "portion_temp": "temperatuur/portie",
        "pricing_value": "prijs/waarde",
        "ambience": "sfeer",
        "order_accuracy": "bestelnauwkeurigheid",
        "cleanliness": "schoonmaak/hygiëne",
    }
    items = sorted(((nl_map.get(k, k), int(v)) for k, v in counts.items()), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(items, columns=["categorie", "aantal"]) 
    df.to_csv(OUTPUT_DIR / "top_pain_points.csv", index=False)
    if not df.empty:
        top = df.head(top_n)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=top, y="categorie", x="aantal", hue="categorie", ax=ax, palette="rocket")
        if ax.legend_:
            ax.legend_.remove()
        ax.set_title("Meest genoemde pijnpunten (negatieve reviews)")
        ax.set_xlabel("Aantal vermeldingen")
        ax.set_ylabel("")
        plot_and_save(fig, "top_pain_points.png")


def export_average_rating(avg_rating: float) -> None:
    """Exporteer totale gemiddelde beoordeling als CSV en eenvoudige afbeelding (Nederlands)."""
    ensure_output_dir()
    pd.DataFrame({"gemiddelde_beoordeling": [avg_rating]}).to_csv(OUTPUT_DIR / "average_rating.csv", index=False)

    # Simple image showing the average rating number
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis('off')
    ax.text(0.5, 0.6, f"{avg_rating:.2f}", ha='center', va='center', fontsize=36, fontweight='bold')
    ax.text(0.5, 0.3, "Gemiddelde beoordeling (1-5)", ha='center', va='center', fontsize=12)
    plot_and_save(fig, "average_rating.png")


def export_sentiment_aggregate(labels) -> None:
    """Exporteer sentiment totalen als CSV (Nederlands)."""
    ensure_output_dir()
    # Map Engelse labels naar Nederlands
    def nl_label(x: str) -> str:
        x = str(x).lower()
        return {"negative": "negatief", "neutral": "neutraal", "positive": "positief"}.get(x, x)

    s = pd.Series([nl_label(x) for x in labels], name="sentiment").value_counts().rename_axis("sentiment").reset_index(name="aantal")
    s.to_csv(OUTPUT_DIR / "sentiment_counts.csv", index=False)
