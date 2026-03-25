# Character-Level Name Generation using RNN Variants

> **Assignment Problem 2** — Character-level Indian name generation  
> Models: Vanilla RNN · Bidirectional LSTM · RNN with Attention

---

## Project Structure

```
.
├── TrainingNames.txt          # 1 076 Indian names (Task 0 dataset)
├── generate_dataset.py        # Task 0 – regenerate dataset via Claude API
│
├── dataset.py                 # Dataset class, vocab, encode/decode utilities
├── models.py                  # Task 1 – VanillaRNN / BLSTM / AttentionRNN
├── train.py                   # Training loop, checkpointing, early stopping
├── generate_names.py          # Autoregressive name generation (temperature sampling)
├── evaluate.py                # Task 2 – novelty rate, diversity, bonus metrics
├── plot_results.py            # Loss curves + metric bar charts
├── main.py                    # End-to-end pipeline (train → generate → evaluate)
│
├── requirements.txt
├── REPORT.md                  # Full written report (Tasks 1–3)
│
├── checkpoints/               # Saved best-model weights  (created at runtime)
├── generated/                 # Generated name text files (created at runtime)
└── results/                   # JSON results + plots       (created at runtime)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> PyTorch with CUDA is optional but speeds up training significantly.  
> CPU training completes in ~5–10 min per model for 100 epochs.

### 2. (Optional) Regenerate the dataset via LLM

The repository already ships with `TrainingNames.txt` (1 076 names).  
To regenerate it using the Claude API:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python generate_dataset.py           # writes TrainingNames.txt
python generate_dataset.py --target 1000 --out MyNames.txt   # custom path
```

### 3. Train all three models

```bash
python main.py
```

This will:
1. Train **VanillaRNN**, **BLSTM**, and **AttentionRNN** for up to 100 epochs each
2. Generate 200 names per model (temperature = 0.8)
3. Print a comparison table of novelty rate, diversity, etc.
4. Save everything to `checkpoints/`, `generated/`, and `results/`

Common options:

```bash
python main.py --epochs 150 --n_gen 300 --temperature 0.9
python main.py --skip_train          # generate + evaluate only (needs checkpoints)
```

### 4. Train a single model

```bash
python train.py --model VanillaRNN --epochs 100 --lr 1e-3
python train.py --model BLSTM
python train.py --model AttentionRNN
```

### 5. Generate names

```bash
# From all trained models
python generate_names.py --n 200 --temperature 0.8

# From one model
python generate_names.py --model BLSTM --n 100 --temperature 0.7
```

Output is saved to `generated/<ModelName>_generated.txt`.

### 6. Evaluate

```bash
python evaluate.py        # reads generated/ and TrainingNames.txt automatically
```

### 7. Plot results

```bash
python plot_results.py    # saves results/loss_curves.png and metrics_comparison.png
```

---

## Model Architectures at a Glance

| Model | Architecture | Parameters* |
|---|---|---|
| VanillaRNN | Embed(64) → RNN(256, 2L) → Linear | ~232 K |
| BLSTM | Embed(64) → BiLSTM(128×2, 2L) → Linear | ~612 K |
| AttentionRNN | Embed(64) → RNN(256, 2L) → Attention → Linear | ~312 K |

\* With this dataset, vocab size ≈ **55** (shown by `main.py`); exact trainable counts:
VanillaRNN **231 671**, BLSTM **611 575**, AttentionRNN **311 799**.

---

## Shared Hyperparameters

| Hyperparameter | Value |
|---|---|
| Embedding dimension | 64 |
| Number of layers | 2 |
| Dropout | 0.3 |
| Optimiser | Adam |
| Learning rate | 1 × 10⁻³ |
| LR scheduler | ReduceLROnPlateau (patience=5, ×0.5) |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping patience | 15 |
| Gradient clipping | max norm = 1.0 |

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Novelty Rate** | (novel names / total) × 100 | % of names not seen in training |
| **Diversity** | unique names / total names | 1 = all different, 0 = all same |
| Avg Length | mean(len(name)) | — |
| Char Coverage | train\_chars ∩ gen\_chars / train\_chars | phonological coverage |

---

## Example results (200 names, temperature 0.8)

Concrete numbers from a trained run (`results/evaluation_results.json`):

| Model | Novelty | Diversity |
|---|---|---|
| VanillaRNN | **92.0%** | **0.975** |
| BLSTM | **90.5%** | **0.985** |
| AttentionRNN | **92.0%** | **0.990** |

BLSTM uses **prefix-aligned training** in `train.py` and prefix re-encoding in
`generate_names.py`. AttentionRNN uses **causal prefix attention** in `models.py`
and the same **prefix re-encoding** when sampling.

Metrics in `evaluation_results.json` reflect whatever was last written by `main.py`
or `evaluate.py`. If you run `generate_names.py` again, novelty/diversity can
shift slightly because sampling is random—re-run `evaluate.py` afterward for a
single consistent table.

See `REPORT.md` for qualitative analysis and failure modes.

---

## File-by-File Description

| File | Description |
|---|---|
| `dataset.py` | `NameDataset` (PyTorch Dataset), `collate_fn` for batching |
| `models.py` | `VanillaRNN`, `BidirectionalLSTM`, `AttentionRNN`, `AdditiveAttention` |
| `train.py` | `train_model()` with validation, checkpointing, early stopping |
| `generate_names.py` | `generate_name()` and `generate_names_batch()` with temperature |
| `evaluate.py` | `novelty_rate()`, `diversity()`, `char_coverage()`, comparison table |
| `plot_results.py` | Matplotlib loss curves and metric bar charts |
| `main.py` | Top-level pipeline runner |
| `generate_dataset.py` | Claude API calls to produce `TrainingNames.txt` |

---

## Tips

- **Temperature**: 0.6–0.7 for realistic names, 0.8–0.9 for more creative ones, >1.0 for chaotic.
- **GPU**: Set `CUDA_VISIBLE_DEVICES=0` to use a GPU; training is ~10× faster.
- **Minimum name length**: Filter `len(name) < 3` post-generation to remove noise.
- **Reproducibility**: Pass `--seed 42` (extend `main.py`) and `torch.manual_seed(42)` for deterministic runs.
