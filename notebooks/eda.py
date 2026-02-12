"""
eda.py — Exploratory Data Analysis for the Sport vs Politics dataset.
Generates class-distribution charts, text-length histograms, word clouds,
and top-N unigram/bigram frequency plots.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import load_data  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def class_distribution(df):
    """Bar chart of class counts."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["category"].value_counts()
    colors = ["#2196F3", "#FF9800"]
    counts.plot.bar(ax=ax, color=colors, edgecolor="black")
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Documents")
    for i, v in enumerate(counts):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=150)
    plt.close()
    print("[INFO] Saved class_distribution.png")


def text_length_histogram(df):
    """Histogram of document lengths per class."""
    df["word_count"] = df["clean_text"].apply(lambda x: len(str(x).split()))
    fig, ax = plt.subplots(figsize=(8, 5))
    for cat, color in zip(["sport", "politics"], ["#2196F3", "#FF9800"]):
        subset = df[df["category"] == cat]
        ax.hist(subset["word_count"], bins=30, alpha=0.65, label=cat.title(), color=color, edgecolor="black")
    ax.set_title("Document Length Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Words (after cleaning)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "text_length_hist.png"), dpi=150)
    plt.close()
    print("[INFO] Saved text_length_hist.png")

    # Print stats
    print("\n--- Text Length Statistics ---")
    print(df.groupby("category")["word_count"].describe().round(1))


def word_clouds(df):
    """One word cloud per class."""
    for cat, cmap in zip(["sport", "politics"], ["Blues", "Oranges"]):
        text = " ".join(df[df["category"] == cat]["clean_text"].tolist())
        wc = WordCloud(width=800, height=400, background_color="white",
                       colormap=cmap, max_words=150).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud — {cat.title()}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"wordcloud_{cat}.png"), dpi=150)
        plt.close()
    print("[INFO] Saved word clouds")


def top_ngrams(df, n=1, top_k=20):
    """Bar charts of top-k n-grams per class."""
    label = "Unigrams" if n == 1 else "Bigrams"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (cat, color) in enumerate(zip(["sport", "politics"], ["#2196F3", "#FF9800"])):
        texts = df[df["category"] == cat]["clean_text"].tolist()
        if n == 1:
            tokens = " ".join(texts).split()
            counter = Counter(tokens)
        else:
            tokens = []
            for t in texts:
                words = t.split()
                tokens.extend(zip(words, words[1:]))
            counter = Counter([" ".join(bg) for bg in tokens])
        most_common = counter.most_common(top_k)
        words, counts = zip(*most_common)
        axes[idx].barh(range(top_k), counts[::-1], color=color, edgecolor="black")
        axes[idx].set_yticks(range(top_k))
        axes[idx].set_yticklabels(list(words)[::-1])
        axes[idx].set_title(f"Top {top_k} {label} — {cat.title()}", fontweight="bold")
        axes[idx].set_xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"top_{label.lower()}.png"), dpi=150)
    plt.close()
    print(f"[INFO] Saved top_{label.lower()}.png")


if __name__ == "__main__":
    _, _, _, _, df = load_data()
    class_distribution(df)
    text_length_histogram(df)
    word_clouds(df)
    top_ngrams(df, n=1)
    top_ngrams(df, n=2)
    print("\n[DONE] All EDA figures generated.")
