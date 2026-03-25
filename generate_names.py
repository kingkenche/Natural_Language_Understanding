"""
generate_names.py
-----------------
Autoregressive character-level name generation with temperature sampling.

Usage (standalone)
------------------
    python generate_names.py --model VanillaRNN --n 100 --temperature 0.8
    python generate_names.py --model all --n 200 --out generated/
"""

import argparse
import os

import torch
import torch.nn.functional as F

from dataset import NameDataset, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from models import VanillaRNN, BidirectionalLSTM, AttentionRNN


# ──────────────────────────────────────────────────────────────────────
# Core generation function
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_name(
    model,
    dataset     : NameDataset,
    device,
    temperature : float = 0.8,
    max_len     : int   = 25,
) -> str:
    """
    Generate one name using autoregressive (token-by-token) sampling.

    Strategy
    --------
    Vanilla RNN: step with ``(1, 1)`` input and carry hidden state.

    BiLSTM and AttentionRNN: each step append to a growing token list and
    forward the **full prefix** (same mask as causal training / attention).

    Temperature
    -----------
    • temperature → 0 : greedy / sharp (repetitive)
    • temperature = 1 : vanilla sampling
    • temperature > 1 : more random / creative
    """
    model.eval()

    sos_idx = dataset.char2idx[SOS_TOKEN]
    eos_idx = dataset.char2idx[EOS_TOKEN]
    pad_idx = dataset.char2idx[PAD_TOKEN]

    # BiLSTM & AttentionRNN: re-run on the full prefix each step so the model
    # sees the same context as in training (no missing history in rnn_out).
    if isinstance(model, (BidirectionalLSTM, AttentionRNN)):
        seq = [sos_idx]
        generated = []
        for _ in range(max_len):
            inp = torch.tensor([seq], dtype=torch.long, device=device)
            logits, _ = model(inp, None)
            logits_step = logits[:, -1, :] / temperature
            probs         = F.softmax(logits_step, dim=-1)
            next_idx      = torch.multinomial(probs, num_samples=1)  # (1, 1)
            idx = next_idx.item()
            if idx == eos_idx:
                break
            ch = dataset.idx2char[idx]
            if ch not in (PAD_TOKEN, SOS_TOKEN):
                generated.append(ch)
            seq.append(idx)
        return "".join(generated)

    # Seed the model with <SOS>
    inp    = torch.tensor([[sos_idx]], dtype=torch.long, device=device)  # (1, 1)
    hidden = model.init_hidden(1, device)

    generated = []

    for _ in range(max_len):
        logits, hidden = model(inp, hidden)          # logits: (1, 1, V)
        logits_step    = logits[:, -1, :] / temperature  # (1, V)
        probs          = F.softmax(logits_step, dim=-1)
        next_idx       = torch.multinomial(probs, num_samples=1)  # (1, 1)

        idx = next_idx.item()
        if idx == eos_idx:
            break
        ch = dataset.idx2char[idx]
        if ch not in (PAD_TOKEN, SOS_TOKEN):
            generated.append(ch)

        inp = next_idx  # feed the predicted token back (1, 1)

    return "".join(generated)


def generate_names_batch(
    model,
    dataset     : NameDataset,
    device,
    n           : int   = 200,
    temperature : float = 0.8,
) -> list:
    """Generate *n* names and return them as a list of strings."""
    names = []
    for _ in range(n):
        name = generate_name(model, dataset, device, temperature)
        if name:                     # skip empty strings
            names.append(name)
    return names


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

MODEL_CLASSES = {
    "VanillaRNN"  : VanillaRNN,
    "BLSTM"       : BidirectionalLSTM,
    "AttentionRNN": AttentionRNN,
}

MODEL_KWARGS = {
    "VanillaRNN"  : dict(embed_dim=64, hidden_size=256, num_layers=2, dropout=0.3),
    "BLSTM"       : dict(embed_dim=64, hidden_size=128, num_layers=2, dropout=0.3),
    "AttentionRNN": dict(embed_dim=64, hidden_size=256, num_layers=2, dropout=0.3),
}


def load_model(model_name: str, vocab_size: int, checkpoint_dir: str, device):
    cls    = MODEL_CLASSES[model_name]
    kwargs = MODEL_KWARGS[model_name]
    model  = cls(vocab_size=vocab_size, **kwargs).to(device)
    ckpt   = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"No checkpoint found at '{ckpt}'. "
            "Please run train.py first."
        )
    state = torch.load(ckpt, map_location=device)
    if model_name == "BLSTM":
        # Allow extra keys from older checkpoints (e.g. removed helper modules).
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate Indian names from trained models")
    parser.add_argument("--data",        default="TrainingNames.txt")
    parser.add_argument("--model",       default="all",
                        choices=["all", "VanillaRNN", "BLSTM", "AttentionRNN"])
    parser.add_argument("--n",           type=int,   default=200, help="Names to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--out",         default="generated",   help="Output directory")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NameDataset(args.data)
    os.makedirs(args.out, exist_ok=True)

    names_to_run = (
        list(MODEL_CLASSES.keys()) if args.model == "all" else [args.model]
    )

    for mname in names_to_run:
        print(f"\nGenerating {args.n} names with {mname}  (T={args.temperature}) …")
        try:
            model = load_model(mname, dataset.vocab_size, args.checkpoint_dir, device)
        except FileNotFoundError as exc:
            print(f"  ✗ {exc}")
            continue

        generated = generate_names_batch(model, dataset, device, args.n, args.temperature)

        out_path = os.path.join(args.out, f"{mname}_generated.txt")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(generated))

        print(f"  ✓ Saved {len(generated)} names → {out_path}")
        print(f"  Sample : {generated[:10]}")


if __name__ == "__main__":
    main()
