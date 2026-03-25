"""
03b_train_scratch.py
─────────────────────────────────────────────────────────────────────────────
TASK 2 (Extension) – WORD2VEC FROM SCRATCH  (NumPy only, zero ML libraries)

Implements the full Word2Vec training loop by hand:
  • Vocabulary building with min-count filtering
  • Subsampling of frequent words  (Mikolov 2013, §2.3)
  • Negative-sampling table        (unigram distribution ^ 0.75)
  • CBOW  forward + backward pass  (manual gradient derivation)
  • Skip-gram forward + backward   (manual gradient derivation)
  • Linear learning-rate decay (matches Gensim default schedule)

Two models are trained to match the Gensim Experiment-2 config
(dim=100, window=5, neg=10, 10 epochs) so they can be compared fairly.

Saved artefacts
  models/CBOW_scratch.npz          – W_in, W_out, word2idx, idx2word
  models/SkipGram_scratch.npz      – same
  outputs/scratch_loss_curve.png   – per-epoch loss for both architectures

Usage:
    python 03b_train_scratch.py
─────────────────────────────────────────────────────────────────────────────
"""

import json, time, logging, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE      = Path(__file__).resolve().parent
SENT_FILE = BASE / "corpus" / "sentences.json"
MODEL_DIR = BASE / "models"
OUT_DIR   = BASE / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ── Hyperparameters (matching Gensim Experiment-2) ────────────────────────
EMBED_DIM    = 100
WINDOW       = 5
NEG_SAMPLES  = 10
EPOCHS       = 10          # scratch is pure Python so fewer epochs; quality converges
ALPHA_START  = 0.025       # initial learning rate
ALPHA_MIN    = 0.0001      # minimum learning rate (linear decay)
MIN_COUNT    = 3           # discard words appearing < MIN_COUNT times
SUBSAMPLE_T  = 1e-3        # subsampling threshold (same as Gensim default)
NS_EXP       = 0.75        # exponent for unigram distribution
NS_TABLE_SZ  = 10_000_000  # negative-sampling table size


# ══════════════════════════════════════════════════════════════════════════
# 1.  VOCABULARY
# ══════════════════════════════════════════════════════════════════════════

class Vocabulary:
    """Builds vocabulary from tokenised sentences with min-count filtering."""

    def __init__(self, sentences: list[list[str]], min_count: int = MIN_COUNT):
        log.info("  Building vocabulary …")
        freq = collections.Counter(w for s in sentences for w in s)

        # Keep only words meeting min_count
        kept = {w: c for w, c in freq.items() if c >= min_count}
        # Sort by frequency descending for deterministic indexing
        vocab_sorted = sorted(kept.items(), key=lambda x: -x[1])

        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str]      = []
        self.freq:     np.ndarray     = np.zeros(len(vocab_sorted), dtype=np.float32)

        for i, (w, c) in enumerate(vocab_sorted):
            self.word2idx[w] = i
            self.idx2word.append(w)
            self.freq[i] = c

        self.size        = len(self.idx2word)
        self.total_words = int(self.freq.sum())

        # Subsampling probability: p_keep(w) = sqrt(t / f(w))  (clamped to 1)
        f_ratio           = self.freq / self.total_words
        self.keep_prob    = np.minimum(
            np.sqrt(SUBSAMPLE_T / (f_ratio + 1e-10)), 1.0
        ).astype(np.float32)

        log.info(f"  Vocabulary: {self.size:,} words  "
                 f"(from {len(freq):,} raw, min_count={min_count})")

    def subsample(self, sentence: list[str]) -> list[int]:
        """Convert a token list to indices, dropping frequent words randomly."""
        indices = []
        for w in sentence:
            idx = self.word2idx.get(w)
            if idx is None:
                continue
            if np.random.random() < self.keep_prob[idx]:
                indices.append(idx)
        return indices


# ══════════════════════════════════════════════════════════════════════════
# 2.  NEGATIVE SAMPLER
# ══════════════════════════════════════════════════════════════════════════

class NegativeSampler:
    """
    Negative-sampling table based on the unigram distribution raised to
    the 3/4 power (Mikolov 2013).  A pre-built lookup table is used for
    fast O(1) sampling.
    """

    def __init__(self, vocab: Vocabulary, table_size: int = NS_TABLE_SZ):
        log.info("  Building negative-sampling table …")
        counts_pow = vocab.freq ** NS_EXP
        probs      = counts_pow / counts_pow.sum()
        # Draw table_size word indices according to unigram^0.75 distribution
        self._table = np.random.choice(vocab.size, size=table_size, p=probs)
        self._ptr   = 0
        log.info(f"  Negative-sampling table built ({table_size:,} entries)")

    def sample(self, k: int, exclude: set) -> list[int]:
        """Return k negative-sample indices, skipping words in `exclude`."""
        negs = []
        while len(negs) < k:
            idx = int(self._table[self._ptr % len(self._table)])
            self._ptr += 1
            if idx not in exclude:
                negs.append(idx)
        return negs


# ══════════════════════════════════════════════════════════════════════════
# 3.  MODEL (weight matrices + gradient updates)
# ══════════════════════════════════════════════════════════════════════════

class Word2VecScratch:
    """
    Weight matrices and manual gradient updates for both CBOW and Skip-gram.

    W_in  shape (V, d) – input  embeddings (the final word vectors we use)
    W_out shape (V, d) – output embeddings (discarded after training)

    Both are initialised as in the original paper:
      W_in  ~ Uniform(-0.5/d, 0.5/d)
      W_out = 0
    """

    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
        rng        = np.random.default_rng(seed)
        self.V     = vocab_size
        self.d     = embed_dim
        self.W_in  = rng.uniform(-0.5 / embed_dim, 0.5 / embed_dim,
                                 (vocab_size, embed_dim)).astype(np.float32)
        self.W_out = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    # ── numerically stable sigmoid ────────────────────────────────────────
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    # ── core negative-sampling update (shared by CBOW and Skip-gram) ──────
    def _ns_update(self,
                   h:          np.ndarray,   # hidden vector  (d,)
                   target_idx: int,
                   neg_indices: list[int],
                   lr:         float
                   ) -> tuple[float, np.ndarray]:
        """
        Compute negative-sampling loss and update W_out.
        Returns (scalar loss, gradient w.r.t. h).

        Loss = -log σ(W_out[t]·h)  –  Σ_i log σ(–W_out[n_i]·h)

        Gradients:
          ∂L/∂W_out[t]   = (σ(score_t) – 1) · h
          ∂L/∂W_out[n_i] =  σ(score_ni)     · h
          ∂L/∂h          = (σ(score_t) – 1) · W_out[t]
                           + Σ_i σ(score_ni) · W_out[n_i]
        """
        # ── positive sample ───────────────────────────────────────────────
        score_t  = float(h @ self.W_out[target_idx])
        sig_t    = float(self._sigmoid(score_t))
        loss     = -np.log(sig_t + 1e-10)

        err_t    = sig_t - 1.0                         # scalar gradient signal
        grad_h   = err_t * self.W_out[target_idx]      # (d,)  ∂L/∂h from +ve
        self.W_out[target_idx] -= lr * err_t * h

        # ── negative samples (vectorised over k) ─────────────────────────
        neg_arr   = np.array(neg_indices, dtype=np.int32)
        W_neg     = self.W_out[neg_arr]                # (k, d)
        scores_n  = W_neg @ h                          # (k,)
        sig_n     = self._sigmoid(scores_n)            # (k,)
        loss     -= float(np.sum(np.log(1.0 - sig_n + 1e-10)))

        # gradient of W_out for negative words: sig_n * h
        self.W_out[neg_arr] -= lr * sig_n[:, None] * h[None, :]  # (k,d)
        # gradient of h from negative words
        grad_h += sig_n @ W_neg                        # (d,)

        return loss, grad_h

    # ── Skip-gram training step ───────────────────────────────────────────
    def train_skipgram(self,
                       center_idx:  int,
                       context_idx: int,
                       neg_indices: list[int],
                       lr:          float) -> float:
        """
        Skip-gram: given center word, predict one context word.

        Forward:
          h = W_in[center]            (the input embedding)
          Minimise NS loss w.r.t. context_idx as target.

        Backward:
          ∂L/∂W_in[center] = ∂L/∂h   (direct, no averaging)
        """
        h              = self.W_in[center_idx].copy()
        loss, grad_h   = self._ns_update(h, context_idx, neg_indices, lr)
        self.W_in[center_idx] -= lr * grad_h
        return loss

    # ── CBOW training step ────────────────────────────────────────────────
    def train_cbow(self,
                   context_indices: list[int],
                   target_idx:      int,
                   neg_indices:     list[int],
                   lr:              float) -> float:
        """
        CBOW: given context words, predict center word.

        Forward:
          h = (1/C) Σ_c W_in[c]      (mean of context embeddings)
          Minimise NS loss w.r.t. target_idx.

        Backward:
          ∂L/∂W_in[c] = (1/C) · ∂L/∂h   for each context word c
        """
        if not context_indices:
            return 0.0
        C              = len(context_indices)
        h              = self.W_in[context_indices].mean(axis=0)   # (d,)
        loss, grad_h   = self._ns_update(h, target_idx, neg_indices, lr)
        # Distribute gradient equally to all context words
        self.W_in[context_indices] -= (lr / C) * grad_h[None, :]
        return loss

    # ── cosine-similarity nearest neighbours ─────────────────────────────
    def most_similar(self, word: str, vocab: Vocabulary, topn: int = 5):
        if word not in vocab.word2idx:
            return []
        idx      = vocab.word2idx[word]
        vec      = self.W_in[idx]
        norms    = np.linalg.norm(self.W_in, axis=1) + 1e-10   # (V,)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
        sims     = (self.W_in / norms[:, None]) @ vec_norm      # (V,)
        sims[idx] = -2.0                                        # exclude self
        top_idx  = np.argsort(sims)[::-1][:topn]
        return [(vocab.idx2word[i], float(sims[i])) for i in top_idx]

    # ── analogy: positive1 + positive2 – negative1 ───────────────────────
    def most_similar_analogy(self,
                             positive: list[str],
                             negative: list[str],
                             vocab:    Vocabulary,
                             topn:     int = 5):
        pos_ok = [w for w in positive if w in vocab.word2idx]
        neg_ok = [w for w in negative if w in vocab.word2idx]
        if not pos_ok:
            return []
        query  = sum(self.W_in[vocab.word2idx[w]] for w in pos_ok)
        query -= sum(self.W_in[vocab.word2idx[w]] for w in neg_ok)
        exclude = {vocab.word2idx[w] for w in pos_ok + neg_ok
                   if w in vocab.word2idx}
        norms   = np.linalg.norm(self.W_in, axis=1) + 1e-10
        q_norm  = query / (np.linalg.norm(query) + 1e-10)
        sims    = (self.W_in / norms[:, None]) @ q_norm
        for idx in exclude:
            sims[idx] = -2.0
        top_idx = np.argsort(sims)[::-1][:topn]
        return [(vocab.idx2word[i], float(sims[i])) for i in top_idx]

    # ── save / load ───────────────────────────────────────────────────────
    def save(self, path: Path, vocab: Vocabulary, meta: dict):
        np.savez_compressed(
            path,
            W_in     = self.W_in,
            W_out    = self.W_out,
            idx2word = np.array(vocab.idx2word),
        )
        json_path = path.with_suffix(".json")
        json_path.write_text(json.dumps({
            **meta,
            "word2idx": vocab.word2idx,
        }, indent=2))
        log.info(f"  Saved → {path}")

    @classmethod
    def load(cls, path: Path) -> tuple["Word2VecScratch", "Vocabulary"]:
        data       = np.load(path, allow_pickle=True)
        idx2word   = list(data["idx2word"])
        word2idx   = {w: i for i, w in enumerate(idx2word)}

        model      = cls.__new__(cls)
        model.W_in  = data["W_in"]
        model.W_out = data["W_out"]
        model.V, model.d = model.W_in.shape

        vocab           = Vocabulary.__new__(Vocabulary)
        vocab.idx2word  = idx2word
        vocab.word2idx  = word2idx
        vocab.size      = len(idx2word)
        vocab.freq      = np.ones(vocab.size, dtype=np.float32)   # placeholder
        vocab.keep_prob = np.ones(vocab.size, dtype=np.float32)

        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            meta      = json.loads(meta_path.read_text())
            model.tag = meta.get("tag", "unknown")
        else:
            model.tag = path.stem

        return model, vocab


# ══════════════════════════════════════════════════════════════════════════
# 4.  TRAINING LOOPS
# ══════════════════════════════════════════════════════════════════════════

def _lr(alpha_start: float, alpha_min: float,
        step: int, total_steps: int) -> float:
    """Linear learning-rate decay (identical to Gensim)."""
    progress = step / max(total_steps, 1)
    return max(alpha_start * (1.0 - progress), alpha_min)


def train_skipgram(model:     Word2VecScratch,
                   sentences: list[list[str]],
                   vocab:     Vocabulary,
                   sampler:   NegativeSampler,
                   epochs:    int,
                   window:    int,
                   neg_k:     int,
                   alpha:     float = ALPHA_START,
                   ) -> list[float]:
    """
    Full Skip-gram training over `epochs` passes.
    Returns per-epoch mean loss (for plotting).
    """
    epoch_losses: list[float] = []

    # Pre-count total training pairs for LR decay
    total_steps = 0
    for sent in sentences:
        idxs = vocab.subsample(sent)
        total_steps += sum(
            min(pos, window) + min(len(idxs) - 1 - pos, window)
            for pos in range(len(idxs))
        )
    total_steps *= epochs
    step = 0

    for epoch in range(epochs):
        epoch_loss, epoch_pairs = 0.0, 0
        for sent in sentences:
            idxs = vocab.subsample(sent)
            if len(idxs) < 2:
                continue
            for pos, center_idx in enumerate(idxs):
                # Dynamic window: sample actual window size each time
                dyn_win = np.random.randint(1, window + 1)
                ctx_start = max(0, pos - dyn_win)
                ctx_end   = min(len(idxs), pos + dyn_win + 1)

                for ctx_pos in range(ctx_start, ctx_end):
                    if ctx_pos == pos:
                        continue
                    ctx_idx  = idxs[ctx_pos]
                    negs     = sampler.sample(neg_k, exclude={center_idx, ctx_idx})
                    lr_now   = _lr(alpha, ALPHA_MIN, step, total_steps)
                    loss     = model.train_skipgram(center_idx, ctx_idx, negs, lr_now)
                    epoch_loss  += loss
                    epoch_pairs += 1
                    step        += 1

        mean_loss = epoch_loss / max(epoch_pairs, 1)
        epoch_losses.append(mean_loss)
        log.info(f"    Epoch {epoch+1}/{epochs}  loss={mean_loss:.4f}  "
                 f"lr={_lr(alpha, ALPHA_MIN, step, total_steps):.5f}  "
                 f"pairs={epoch_pairs:,}")

    return epoch_losses


def train_cbow(model:     Word2VecScratch,
               sentences: list[list[str]],
               vocab:     Vocabulary,
               sampler:   NegativeSampler,
               epochs:    int,
               window:    int,
               neg_k:     int,
               alpha:     float = ALPHA_START,
               ) -> list[float]:
    """
    Full CBOW training over `epochs` passes.
    Returns per-epoch mean loss (for plotting).
    """
    epoch_losses: list[float] = []

    total_steps = sum(
        len(vocab.subsample(s)) for s in sentences
    ) * epochs
    step = 0

    for epoch in range(epochs):
        epoch_loss, epoch_pairs = 0.0, 0
        for sent in sentences:
            idxs = vocab.subsample(sent)
            if len(idxs) < 2:
                continue
            for pos, target_idx in enumerate(idxs):
                dyn_win = np.random.randint(1, window + 1)
                ctx_start = max(0, pos - dyn_win)
                ctx_end   = min(len(idxs), pos + dyn_win + 1)
                ctx_idxs  = [idxs[i] for i in range(ctx_start, ctx_end)
                             if i != pos]
                if not ctx_idxs:
                    continue

                negs     = sampler.sample(neg_k, exclude={target_idx} | set(ctx_idxs))
                lr_now   = _lr(alpha, ALPHA_MIN, step, total_steps)
                loss     = model.train_cbow(ctx_idxs, target_idx, negs, lr_now)
                epoch_loss  += loss
                epoch_pairs += 1
                step        += 1

        mean_loss = epoch_loss / max(epoch_pairs, 1)
        epoch_losses.append(mean_loss)
        log.info(f"    Epoch {epoch+1}/{epochs}  loss={mean_loss:.4f}  "
                 f"lr={_lr(alpha, ALPHA_MIN, step, total_steps):.5f}  "
                 f"pairs={epoch_pairs:,}")

    return epoch_losses


# ══════════════════════════════════════════════════════════════════════════
# 5.  LOSS CURVE PLOT
# ══════════════════════════════════════════════════════════════════════════

def plot_loss_curves(cbow_losses: list[float],
                     sg_losses:   list[float]):
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs_x = list(range(1, len(cbow_losses) + 1))
    ax.plot(epochs_x, cbow_losses,    marker="o", color="#3498db",
            label="CBOW (scratch)",     linewidth=2)
    ax.plot(epochs_x, sg_losses,      marker="s", color="#e74c3c",
            label="Skip-gram (scratch)", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean NS Loss", fontsize=12)
    ax.set_title("From-Scratch Word2Vec Training Loss\n"
                 "(dim=100, window=5, neg=10, IIT Jodhpur corpus)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "scratch_loss_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Loss curve saved → {out}")


# ══════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main() -> dict:
    print("=" * 60)
    print("  TASK 2b – WORD2VEC FROM SCRATCH  (NumPy only)")
    print("=" * 60)

    if not SENT_FILE.exists():
        print("  ERROR: Run 02_preprocess.py first.")
        return {}

    with open(SENT_FILE) as f:
        sentences: list[list[str]] = json.load(f)
    print(f"\n  Loaded {len(sentences):,} sentences.")

    # ── Build shared vocabulary and negative sampler ──────────────────────
    vocab   = Vocabulary(sentences, min_count=MIN_COUNT)
    sampler = NegativeSampler(vocab)

    results = {}

    # ── Train CBOW from scratch ───────────────────────────────────────────
    print(f"\n  Training CBOW (scratch)  "
          f"dim={EMBED_DIM}, win={WINDOW}, neg={NEG_SAMPLES}, "
          f"epochs={EPOCHS} …")
    cbow_model = Word2VecScratch(vocab.size, EMBED_DIM)
    t0 = time.time()
    cbow_losses = train_cbow(cbow_model, sentences, vocab, sampler,
                             EPOCHS, WINDOW, NEG_SAMPLES)
    cbow_time = time.time() - t0
    cbow_model.save(MODEL_DIR / "CBOW_scratch.npz", vocab,
                    {"tag": "CBOW_scratch", "arch": "CBOW",
                     "embed_dim": EMBED_DIM, "window": WINDOW,
                     "neg_samples": NEG_SAMPLES, "epochs": EPOCHS,
                     "train_time_s": round(cbow_time, 1)})
    log.info(f"  CBOW scratch done in {cbow_time:.1f}s")
    results["CBOW_scratch"] = {"losses": cbow_losses, "time": cbow_time}

    # ── Train Skip-gram from scratch ──────────────────────────────────────
    print(f"\n  Training Skip-gram (scratch)  "
          f"dim={EMBED_DIM}, win={WINDOW}, neg={NEG_SAMPLES}, "
          f"epochs={EPOCHS} …")
    sg_model = Word2VecScratch(vocab.size, EMBED_DIM)
    t0 = time.time()
    sg_losses = train_skipgram(sg_model, sentences, vocab, sampler,
                               EPOCHS, WINDOW, NEG_SAMPLES)
    sg_time = time.time() - t0
    sg_model.save(MODEL_DIR / "SkipGram_scratch.npz", vocab,
                  {"tag": "SkipGram_scratch", "arch": "SkipGram",
                   "embed_dim": EMBED_DIM, "window": WINDOW,
                   "neg_samples": NEG_SAMPLES, "epochs": EPOCHS,
                   "train_time_s": round(sg_time, 1)})
    log.info(f"  Skip-gram scratch done in {sg_time:.1f}s")
    results["SkipGram_scratch"] = {"losses": sg_losses, "time": sg_time}

    # ── Loss curve ────────────────────────────────────────────────────────
    plot_loss_curves(cbow_losses, sg_losses)

    # ── Quick sanity check ────────────────────────────────────────────────
    print("\n  Quick nearest-neighbour check (CBOW scratch):")
    for word in ["research", "student", "phd", "exam", "department"]:
        nbrs = cbow_model.most_similar(word, vocab, topn=3)
        if nbrs:
            nbr_str = ", ".join(f"{w}({s:.3f})" for w, s in nbrs)
            print(f"    {word:<15} → {nbr_str}")
        else:
            print(f"    {word:<15} → (not in vocabulary)")

    print(f"\n  CBOW scratch   : {cbow_time:.1f}s")
    print(f"  Skip-gram scratch: {sg_time:.1f}s")
    print("\n  Models saved → models/CBOW_scratch.npz, SkipGram_scratch.npz")
    print("  Loss curve  → outputs/scratch_loss_curve.png\n")

    return results


if __name__ == "__main__":
    main()
