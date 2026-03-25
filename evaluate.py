"""
evaluate.py
-----------
Quantitative evaluation of generated name sets.

Metrics
-------
Novelty Rate  : % of generated names that do NOT appear in the training set.
                Higher → the model is more creative / generalising.
Diversity     : |unique generated names| / |total generated names|.
                Higher → less repetition.

Bonus metrics (also computed)
------------------------------
Avg Length       : mean character length of generated names.
Length Std       : standard deviation of generated name lengths.
Char Coverage    : fraction of training vocabulary characters appearing
                   in generated names.

Usage (standalone)
------------------
    python evaluate.py                    # reads generated/ and TrainingNames.txt
    python evaluate.py --generated_dir generated --data TrainingNames.txt
"""

import argparse
import json
import os

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Core metric functions
# ──────────────────────────────────────────────────────────────────────

def novelty_rate(generated: list, training: list) -> float:
    """Return the fraction (0–100) of generated names absent from training."""
    train_set = {n.strip().lower() for n in training}
    novel     = sum(1 for n in generated if n.strip().lower() not in train_set)
    return 100.0 * novel / len(generated) if generated else 0.0


def diversity(generated: list) -> float:
    """Return the fraction of unique names among all generated names."""
    if not generated:
        return 0.0
    return len({n.strip().lower() for n in generated}) / len(generated)


def avg_length(names: list) -> tuple:
    """Return (mean, std) character lengths."""
    if not names:
        return 0.0, 0.0
    lengths = [len(n) for n in names]
    return float(np.mean(lengths)), float(np.std(lengths))


def char_coverage(generated: list, training: list) -> float:
    """Fraction of training characters that appear in generated names."""
    train_chars = set("".join(training).lower())
    gen_chars   = set("".join(generated).lower())
    if not train_chars:
        return 0.0
    return len(gen_chars & train_chars) / len(train_chars)


# ──────────────────────────────────────────────────────────────────────
# Per-model evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_name     : str,
    generated_names: list,
    training_names : list,
    verbose        : bool = True,
) -> dict:
    """Compute all metrics for one model and optionally print a report."""
    nv  = novelty_rate(generated_names, training_names)
    div = diversity(generated_names)
    mu, sigma = avg_length(generated_names)
    cov = char_coverage(generated_names, training_names)

    result = {
        "model"         : model_name,
        "n_generated"   : len(generated_names),
        "novelty_rate"  : round(nv,  2),
        "diversity"     : round(div, 4),
        "avg_length"    : round(mu,  2),
        "length_std"    : round(sigma, 2),
        "char_coverage" : round(cov, 4),
        "samples"       : generated_names[:20],
    }

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Model           : {model_name}")
        print(f"  Names generated : {len(generated_names)}")
        print(f"  Novelty Rate    : {nv:.2f}%")
        print(f"  Diversity       : {div:.4f}")
        print(f"  Avg length      : {mu:.2f} ± {sigma:.2f} chars")
        print(f"  Char coverage   : {cov:.4f}")
        print(f"  Sample names    : {generated_names[:10]}")
        print(f"{'─'*55}")

    return result


# ──────────────────────────────────────────────────────────────────────
# Comparison table
# ──────────────────────────────────────────────────────────────────────

def print_comparison_table(results: list):
    header = (
        f"\n{'='*70}\n"
        f"{'Model':<18} {'Novelty%':>10} {'Diversity':>11} "
        f"{'Avg Len':>9} {'Char Cov':>10}\n"
        f"{'─'*70}"
    )
    print(header)
    for r in results:
        print(
            f"{r['model']:<18} "
            f"{r['novelty_rate']:>9.2f}% "
            f"{r['diversity']:>11.4f} "
            f"{r['avg_length']:>9.2f} "
            f"{r['char_coverage']:>10.4f}"
        )
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def load_names(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated name sets")
    parser.add_argument("--data",          default="TrainingNames.txt")
    parser.add_argument("--generated_dir", default="generated")
    parser.add_argument("--results_dir",   default="results")
    args = parser.parse_args()

    training_names = load_names(args.data)
    os.makedirs(args.results_dir, exist_ok=True)

    results = []
    gen_dir = args.generated_dir

    if not os.path.isdir(gen_dir):
        print(f"Directory '{gen_dir}' not found. Run generate_names.py first.")
        return

    gen_files = sorted(f for f in os.listdir(gen_dir) if f.endswith("_generated.txt"))
    if not gen_files:
        print(f"No '*_generated.txt' files found in '{gen_dir}'.")
        return

    for fname in gen_files:
        model_name     = fname.replace("_generated.txt", "")
        generated_path = os.path.join(gen_dir, fname)
        generated      = load_names(generated_path)
        result         = evaluate_model(model_name, generated, training_names)
        results.append(result)

    print_comparison_table(results)

    out_path = os.path.join(args.results_dir, "evaluation_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
