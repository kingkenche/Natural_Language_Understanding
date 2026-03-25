"""
train.py
--------
Training loop with validation, learning-rate scheduling, and checkpointing
for character-level name generation models.

Usage
-----
    python train.py                         # train all three models
    python train.py --model VanillaRNN      # train a single model
    python train.py --epochs 150 --lr 5e-4
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import NameDataset, collate_fn
from models import VanillaRNN, BidirectionalLSTM, AttentionRNN


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _blstm_prefix_loss(model, x, y, lengths, criterion, device):
    """
    BiLSTM language-model loss aligned with autoregressive *prefix* decoding.

    For each position t, run the BiLSTM on x[b, :t+1] only and predict y[b, t].
    The backward LSTM then never sees tokens that were not available at generation.
    """
    loss_sum = 0.0
    n_step   = 0
    x, y = x.to(device), y.to(device)
    B = x.size(0)
    for b in range(B):
        L = int(lengths[b].item())
        if L <= 0:
            continue
        xb, yb = x[b, :L], y[b, :L]
        for t in range(L):
            prefix = xb[: t + 1].unsqueeze(0)          # (1, t+1)
            logits, _ = model(prefix, None)              # (1, t+1, V)
            logit = logits[0, -1]                      # (V,)
            step  = criterion(logit.unsqueeze(0), yb[t].unsqueeze(0))
            loss_sum += step
            n_step   += 1
    if n_step == 0:
        raise RuntimeError("BLSTM causal loss: no token steps in batch (check data / lengths).")
    return loss_sum / n_step


def train_one_epoch(model, loader, optimizer, criterion, device, clip: float = 1.0):
    model.train()
    total_loss = 0.0
    for x, y, lengths in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        if isinstance(model, BidirectionalLSTM):
            loss = _blstm_prefix_loss(model, x, y, lengths, criterion, device)
        else:
            logits, _ = model(x)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x, y, lengths in loader:
        x, y = x.to(device), y.to(device)
        if isinstance(model, BidirectionalLSTM):
            loss = _blstm_prefix_loss(model, x, y, lengths, criterion, device)
        else:
            logits, _ = model(x)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
        total_loss += loss.item()
    return total_loss / len(loader)


# ──────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────

def train_model(
    model,
    dataset      : NameDataset,
    model_name   : str,
    *,
    epochs       : int   = 100,
    batch_size   : int   = 32,
    lr           : float = 1e-3,
    val_split    : float = 0.1,
    patience     : int   = 10,
    device                = "cpu",
    checkpoint_dir: str  = "checkpoints",
    results_dir   : str  = "results",
) -> dict:
    """
    Train *model* on *dataset* and save the best checkpoint.

    Returns
    -------
    history dict with 'train_loss' and 'val_loss' lists
    """
    # ── split ──────────────────────────────────────────────────────────
    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)

    # ── optimiser / loss ───────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience // 2, factor=0.5
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)   # ignore PAD token

    # ── header ─────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Model      : {model_name}")
    print(f"  Parameters : {model.count_parameters():,}")
    print(f"  Train size : {n_train}   Val size : {n_val}")
    print(f"  Epochs     : {epochs}   LR : {lr}   Batch : {batch_size}")
    print(sep)

    # ── training loop ──────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir,    exist_ok=True)

    best_val  = float("inf")
    no_improve = 0
    history   = {"train_loss": [], "val_loss": []}
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
    t0        = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        # ── checkpoint ─────────────────────────────────────────────────
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        # ── logging ────────────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:4d}/{epochs} | "
                f"Train: {tr_loss:.4f} | Val: {val_loss:.4f} | "
                f"Best: {best_val:.4f} | {elapsed:6.1f}s"
            )

        # ── early stopping ─────────────────────────────────────────────
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"  ✓ Best val loss: {best_val:.4f}  →  saved to {ckpt_path}")

    # ── save loss history ──────────────────────────────────────────────
    hist_path = os.path.join(results_dir, f"{model_name}_history.json")
    with open(hist_path, "w") as fh:
        json.dump(history, fh, indent=2)

    return history


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def build_models(vocab_size: int):
    return {
        "VanillaRNN": VanillaRNN(
            vocab_size  = vocab_size,
            embed_dim   = 64,
            hidden_size = 256,
            num_layers  = 2,
            dropout     = 0.3,
        ),
        "BLSTM": BidirectionalLSTM(
            vocab_size  = vocab_size,
            embed_dim   = 64,
            hidden_size = 128,   # per-direction; effective output = 256
            num_layers  = 2,
            dropout     = 0.3,
        ),
        "AttentionRNN": AttentionRNN(
            vocab_size  = vocab_size,
            embed_dim   = 64,
            hidden_size = 256,
            num_layers  = 2,
            dropout     = 0.3,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Train character-level name generation models")
    parser.add_argument("--data",       default="TrainingNames.txt")
    parser.add_argument("--model",      default="all",
                        choices=["all", "VanillaRNN", "BLSTM", "AttentionRNN"])
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=15)
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = NameDataset(args.data)
    print(f"Dataset: {len(dataset)} names | vocab size: {dataset.vocab_size}")

    models = build_models(dataset.vocab_size)
    to_train = models if args.model == "all" else {args.model: models[args.model]}

    for name, model in to_train.items():
        model = model.to(device)
        train_model(
            model, dataset, name,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            patience    = args.patience,
            device      = device,
        )

    print("\nAll training complete.")


if __name__ == "__main__":
    main()
