import seaborn as sns
from matplotlib import pyplot as plt

from .io_utils import plot_and_save


def run_eda(df):
    sns.set_theme(style="whitegrid")

    # Rating distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="rating", hue="rating", ax=ax, palette="viridis")
    if ax.legend_:
        ax.legend_.remove()
    ax.set_title("Rating distribution")
    plot_and_save(fig, "ratings_distribution.png")

    # Sentiment distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="sentiment", hue="sentiment", order=["negative", "neutral", "positive"], ax=ax, palette=["#d62728", "#ff7f0e", "#2ca02c"]) 
    if ax.legend_:
        ax.legend_.remove()
    ax.set_title("Sentiment distribution")
    plot_and_save(fig, "sentiment_distribution.png")

    # Trend over time (monthly average rating)
    monthly = (
        df.dropna(subset=["timestamp"]).assign(
            month=lambda x: x["timestamp"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
        )
        .groupby("month")["rating"].mean().reset_index()
    )
    if not monthly.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=monthly, x="month", y="rating", marker="o", ax=ax)
        ax.set_title("Average rating over time (monthly)")
        ax.set_ylim(1, 5)
        plot_and_save(fig, "avg_rating_over_time.png")
