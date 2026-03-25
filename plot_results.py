"""
plot_results.py
---------------
Visualise training loss curves and evaluation metrics.

Reads:
    results/<ModelName>_history.json
    results/evaluation_results.json

Produces:
    results/loss_curves.png
    results/metrics_comparison.png

Usage
-----
    python plot_results.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "VanillaRNN"  : "#4C72B0",
    "BLSTM"       : "#DD8452",
    "AttentionRNN": "#55A868",
}


# ──────────────────────────────────────────────────────────────────────
# Loss curves
# ──────────────────────────────────────────────────────────────────────

def plot_loss_curves(results_dir: str = "results"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training & Validation Loss", fontsize=14, fontweight="bold")

    for model_name, color in COLORS.items():
        hist_path = os.path.join(results_dir, f"{model_name}_history.json")
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as fh:
            hist = json.load(fh)

        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], label=model_name, color=color, linewidth=1.8)
        axes[1].plot(epochs, hist["val_loss"],   label=model_name, color=color, linewidth=1.8,
                     linestyle="--")

    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, "loss_curves.png")
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Metrics bar chart
# ──────────────────────────────────────────────────────────────────────

def plot_metrics(results_dir: str = "results"):
    eval_path = os.path.join(results_dir, "evaluation_results.json")
    if not os.path.exists(eval_path):
        print(f"'{eval_path}' not found – skipping metrics plot.")
        return

    with open(eval_path) as fh:
        results = json.load(fh)

    model_names = [r["model"]          for r in results]
    novelty     = [r["novelty_rate"]   for r in results]
    diversity   = [r["diversity"] * 100 for r in results]   # scale to %

    x      = np.arange(len(model_names))
    width  = 0.35
    colors = [COLORS.get(m, "#999") for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Evaluation Metrics – Generated Names", fontsize=14, fontweight="bold")

    # Novelty
    bars = axes[0].bar(x, novelty, width=0.5, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=11)
    axes[0].set_title("Novelty Rate (higher = more creative)")
    axes[0].set_ylabel("Novelty Rate (%)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(model_names, rotation=10)
    axes[0].set_ylim(0, 115)
    axes[0].grid(axis="y", alpha=0.3)

    # Diversity
    bars2 = axes[1].bar(x, diversity, width=0.5, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=11)
    axes[1].set_title("Diversity (higher = less repetition)")
    axes[1].set_ylabel("Diversity (%)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(model_names, rotation=10)
    axes[1].set_ylim(0, 115)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, "metrics_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_loss_curves()
    plot_metrics()
