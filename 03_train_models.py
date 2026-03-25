"""
03_train_models.py
─────────────────────────────────────────────────────────────────────────────
TASK 2 – MODEL TRAINING
Trains 12 Word2Vec variants (CBOW + Skip-gram, across 3 hyperparameter sets)
and saves:
  • All models           → models/
  • Hyperparameter table → outputs/hyperparams_table.png
  • Training summary     → outputs/training_summary.json

Architectures:
  sg=0  →  CBOW   (Continuous Bag of Words)
  sg=1  →  Skip-gram with Negative Sampling  (ns=True by default in gensim)

Experiments:
  Exp   dim   window   neg_samples
  ──────────────────────────────────
   1     50      3          5
   2    100      5         10
   3    200      7         15
─────────────────────────────────────────────────────────────────────────────
"""

import json, time, logging
from pathlib import Path
from gensim.models import Word2Vec
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE      = Path(__file__).resolve().parent
SENT_FILE = BASE / "corpus" / "sentences.json"
MODEL_DIR = BASE / "models"
OUT_DIR   = BASE / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ── Hyperparameter grid ───────────────────────────────────────────────────
EXPERIMENTS = [
    {"exp": 1, "vector_size": 50,  "window": 3, "negative": 5},
    {"exp": 2, "vector_size": 100, "window": 5, "negative": 10},
    {"exp": 3, "vector_size": 200, "window": 7, "negative": 15},
]

ARCHITECTURES = [
    {"sg": 0, "name": "CBOW"},
    {"sg": 1, "name": "SkipGram"},
]

EPOCHS    = 20          # more epochs beneficial for larger corpus
MIN_COUNT = 3           # slightly higher threshold for real corpus
WORKERS   = 4


def train_all(sentences: list[list[str]]) -> list[dict]:
    """Train all 6 models (2 archs × 3 experiments)."""
    results = []

    for arch in ARCHITECTURES:
        for exp in EXPERIMENTS:
            tag    = f"{arch['name']}_e{exp['exp']}"
            params = dict(
                sentences   = sentences,
                vector_size = exp["vector_size"],
                window      = exp["window"],
                negative    = exp["negative"],
                sg          = arch["sg"],
                min_count   = MIN_COUNT,
                workers     = WORKERS,
                epochs      = EPOCHS,
                hs          = 0,      # use negative sampling (not hierarchical softmax)
                seed        = 42,
            )

            logger.info(f"Training {tag}  (dim={exp['vector_size']}, "
                        f"win={exp['window']}, neg={exp['negative']}) …")
            t0 = time.time()
            model = Word2Vec(**params)
            elapsed = time.time() - t0

            model_path = MODEL_DIR / f"{tag}.model"
            model.save(str(model_path))

            vocab_size = len(model.wv)
            logger.info(f"  Done in {elapsed:.1f}s  |  vocab={vocab_size}")

            results.append({
                "tag":         tag,
                "architecture": arch["name"],
                "experiment":  exp["exp"],
                "vector_size": exp["vector_size"],
                "window":      exp["window"],
                "negative":    exp["negative"],
                "epochs":      EPOCHS,
                "vocab_size":  vocab_size,
                "train_time_s": round(elapsed, 2),
                "model_path":  str(model_path),
            })

    return results


def plot_hyperparams_table(results: list[dict]):
    """Render a formatted table of hyperparameters as a figure."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")

    col_labels = [
        "Model", "Architecture", "Exp", "Embedding Dim",
        "Window Size", "Neg Samples", "Epochs", "Vocab Size", "Train Time (s)"
    ]
    rows = []
    for r in results:
        rows.append([
            r["tag"], r["architecture"], r["experiment"],
            r["vector_size"], r["window"], r["negative"],
            r["epochs"], r["vocab_size"], r["train_time_s"],
        ])

    table = ax.table(
        cellText  = rows,
        colLabels = col_labels,
        cellLoc   = "center",
        loc       = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Colour header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for i in range(1, len(rows) + 1):
        colour = "#ecf0f1" if i % 2 == 0 else "#ffffff"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(colour)

    ax.set_title(
        "Word2Vec Model Configurations – IIT Jodhpur Corpus",
        fontsize=14, fontweight="bold", pad=20
    )
    fig.tight_layout()
    out = OUT_DIR / "hyperparams_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Hyperparameter table → {out}")


def main():
    print("=" * 60)
    print("  TASK 2 – MODEL TRAINING")
    print("=" * 60)

    if not SENT_FILE.exists():
        print("  ERROR: Run 02_preprocess.py first.")
        return

    with open(SENT_FILE) as f:
        sentences = json.load(f)
    print(f"\n  Loaded {len(sentences):,} sentences for training.")

    results = train_all(sentences)
    plot_hyperparams_table(results)

    with open(OUT_DIR / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n  All models saved to models/")
    print("  Hyperparameter table saved to outputs/hyperparams_table.png\n")
    return results


if __name__ == "__main__":
    main()
