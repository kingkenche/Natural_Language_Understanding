"""
main.py
-------
End-to-end pipeline:
    1. Load dataset
    2. Train all three models
    3. Generate names from each trained model
    4. Evaluate and print comparison

Usage
-----
    python main.py
    python main.py --epochs 150 --n_gen 200 --temperature 0.8
    python main.py --skip_train          # only generate + evaluate (needs checkpoints)
"""

import argparse
import json
import os

import torch

from dataset       import NameDataset
from models        import VanillaRNN, BidirectionalLSTM, AttentionRNN
from train         import train_model
from generate_names import generate_names_batch, load_model
from evaluate      import evaluate_model, print_comparison_table


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Shared hyper-params
    "embed_dim"  : 64,
    "num_layers" : 2,
    "dropout"    : 0.3,
    "lr"         : 1e-3,
    "batch_size" : 32,
    "patience"   : 15,
    # Per-model hidden sizes (chosen so parameter counts are comparable)
    "hidden_VanillaRNN"  : 256,
    "hidden_BLSTM"       : 128,   # per-direction; output = 256
    "hidden_AttentionRNN": 256,
}


def build_model_registry(vocab_size: int, cfg: dict) -> dict:
    """Return {model_name: model_instance} for all three architectures."""
    return {
        "VanillaRNN": VanillaRNN(
            vocab_size  = vocab_size,
            embed_dim   = cfg["embed_dim"],
            hidden_size = cfg["hidden_VanillaRNN"],
            num_layers  = cfg["num_layers"],
            dropout     = cfg["dropout"],
        ),
        "BLSTM": BidirectionalLSTM(
            vocab_size  = vocab_size,
            embed_dim   = cfg["embed_dim"],
            hidden_size = cfg["hidden_BLSTM"],
            num_layers  = cfg["num_layers"],
            dropout     = cfg["dropout"],
        ),
        "AttentionRNN": AttentionRNN(
            vocab_size  = vocab_size,
            embed_dim   = cfg["embed_dim"],
            hidden_size = cfg["hidden_AttentionRNN"],
            num_layers  = cfg["num_layers"],
            dropout     = cfg["dropout"],
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Name generation pipeline")
    parser.add_argument("--data",        default="TrainingNames.txt")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--n_gen",       type=int,   default=200,
                        help="Number of names to generate per model")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--skip_train",  action="store_true",
                        help="Skip training and use existing checkpoints")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--generated_dir",  default="generated")
    parser.add_argument("--results_dir",    default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Device : {device}")

    # ── 1. Dataset ────────────────────────────────────────────────────
    dataset = NameDataset(args.data)
    print(f"  Dataset : {len(dataset)} names | vocab size : {dataset.vocab_size}")
    print(f"{'='*60}\n")

    cfg     = DEFAULT_CONFIG
    models  = build_model_registry(dataset.vocab_size, cfg)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.generated_dir,  exist_ok=True)
    os.makedirs(args.results_dir,    exist_ok=True)

    # ── 2. Training ───────────────────────────────────────────────────
    if not args.skip_train:
        for name, model in models.items():
            model = model.to(device)
            train_model(
                model, dataset, name,
                epochs      = args.epochs,
                batch_size  = cfg["batch_size"],
                lr          = cfg["lr"],
                patience    = cfg["patience"],
                device      = device,
                checkpoint_dir = args.checkpoint_dir,
                results_dir    = args.results_dir,
            )
    else:
        print("Skipping training – loading existing checkpoints.\n")

    # ── 3. Generate & Evaluate ────────────────────────────────────────
    all_results = []

    for name in models:
        print(f"\nGenerating {args.n_gen} names with '{name}'  (T={args.temperature}) …")
        try:
            model = load_model(name, dataset.vocab_size, args.checkpoint_dir, device)
        except FileNotFoundError as exc:
            print(f"  ✗ {exc}  –  skipping.")
            continue

        generated = generate_names_batch(
            model, dataset, device, n=args.n_gen, temperature=args.temperature
        )

        # Save generated names
        gen_path = os.path.join(args.generated_dir, f"{name}_generated.txt")
        with open(gen_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(generated))

        # Evaluate
        result = evaluate_model(name, generated, dataset.names, verbose=True)
        all_results.append(result)

    # ── 4. Comparison table ───────────────────────────────────────────
    if all_results:
        print_comparison_table(all_results)

    # Save aggregated results
    out_json = os.path.join(args.results_dir, "evaluation_results.json")
    with open(out_json, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nAll results saved → {out_json}")

    # ── 5. Parameter summary ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Model parameter counts (re-instantiated for display)")
    print(f"{'─'*60}")
    for name, model in build_model_registry(dataset.vocab_size, cfg).items():
        print(f"  {name:<20s} : {model.count_parameters():>10,} parameters")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
