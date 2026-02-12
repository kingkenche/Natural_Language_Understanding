"""
evaluate.py — Generate comparison charts and confusion-matrix heatmaps.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")
FIGURES_DIR = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_results():
    path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(path) as f:
        return json.load(f)


def plot_accuracy_comparison(results):
    """Grouped bar chart: Test Accuracy for each (feature, classifier)."""
    features = sorted(set(r["feature"] for r in results),
                      key=["BoW", "TF-IDF", "Bigram TF-IDF"].index)
    classifiers = sorted(set(r["classifier"] for r in results),
                         key=["Logistic Regression", "Multinomial NB", "Linear SVM"].index)

    x = np.arange(len(features))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, clf_name in enumerate(classifiers):
        accs = [
            next(r["test_accuracy"] for r in results
                 if r["feature"] == feat and r["classifier"] == clf_name)
            for feat in features
        ]
        bars = ax.bar(x + i * width, accs, width, label=clf_name, color=colors[i])
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{acc:.2%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Feature Representation", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Test Accuracy — Classifier × Feature Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(features)
    ax.set_ylim(0.80, 1.02)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close()
    print("[INFO] Saved accuracy_comparison.png")


def plot_f1_comparison(results):
    """Grouped bar chart: Macro F1 for each (feature, classifier)."""
    features = sorted(set(r["feature"] for r in results),
                      key=["BoW", "TF-IDF", "Bigram TF-IDF"].index)
    classifiers = sorted(set(r["classifier"] for r in results),
                         key=["Logistic Regression", "Multinomial NB", "Linear SVM"].index)

    x = np.arange(len(features))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, clf_name in enumerate(classifiers):
        f1s = [
            next(r["macro_f1"] for r in results
                 if r["feature"] == feat and r["classifier"] == clf_name)
            for feat in features
        ]
        bars = ax.bar(x + i * width, f1s, width, label=clf_name, color=colors[i])
        for bar, f1 in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{f1:.2%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Feature Representation", fontsize=12)
    ax.set_ylabel("Macro F1-Score", fontsize=12)
    ax.set_title("Macro F1-Score — Classifier × Feature Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(features)
    ax.set_ylim(0.80, 1.02)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "f1_comparison.png"), dpi=150)
    plt.close()
    print("[INFO] Saved f1_comparison.png")


def plot_confusion_matrices(results):
    """One heatmap per experiment."""
    for r in results:
        cm = np.array(r["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Politics", "Sport"],
                    yticklabels=["Politics", "Sport"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        title = f"{r['classifier']} + {r['feature']}"
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        safe_name = title.replace(" ", "_").replace("+", "").replace("-", "")
        plt.savefig(os.path.join(FIGURES_DIR, f"cm_{safe_name}.png"), dpi=120)
        plt.close()
    print("[INFO] Saved confusion-matrix heatmaps")


def plot_cv_comparison(results):
    """Bar chart of CV accuracy with error bars."""
    labels = [f"{r['classifier']}\n{r['feature']}" for r in results]
    means = [r["cv_accuracy_mean"] for r in results]
    stds = [r["cv_accuracy_std"] for r in results]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = sns.color_palette("viridis", len(results))
    bars = ax.bar(range(len(results)), means, yerr=stds, color=colors, capsize=4)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylabel("5-Fold CV Accuracy", fontsize=12)
    ax.set_title("Cross-Validation Accuracy (mean ± std)", fontsize=14, fontweight="bold")
    ax.set_ylim(0.85, 1.02)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cv_accuracy.png"), dpi=150)
    plt.close()
    print("[INFO] Saved cv_accuracy.png")


if __name__ == "__main__":
    results = load_results()
    plot_accuracy_comparison(results)
    plot_f1_comparison(results)
    plot_confusion_matrices(results)
    plot_cv_comparison(results)
    print("[DONE] All evaluation figures generated.")
