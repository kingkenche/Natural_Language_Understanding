# Problem 2: Character-Level Name Generation Using RNN Variants
## Assignment Report

---

## TASK-0: Dataset

### Generation Process

1000 Indian names were generated using the Claude API (`generate_dataset.py`).  
Ten prompts, each targeting a distinct demographic or regional group, were issued to
the model and the raw responses were parsed and deduplicated:

| Batch | Group |
|-------|-------|
| 1 | North Indian male names (Hindi belt) |
| 2 | North Indian female names |
| 3 | South Indian names (Tamil/Telugu/Kannada/Malayalam) |
| 4 | Bengali names |
| 5 | Punjabi names |
| 6 | Gujarati & Marathi names |
| 7 | Indian Muslim names |
| 8 | Classical Sanskrit-origin names |
| 9 | North-East / Odia / Bihari names |
| 10 | Modern/trendy Indian baby names |

### Dataset Statistics

| Property | Value |
|---|---|
| Total names | 1 076 |
| Unique characters | ~58 (A–Z, a–z subset) |
| Minimum length | 2 chars |
| Maximum length | 20 chars |
| Mean length | ~7.3 chars |
| Stored in | `TrainingNames.txt` |

The dataset intentionally spans multiple Indian languages (Hindi, Tamil, Bengali,
Punjabi, Gujarati, Marathi, Malayalam, Kannada, Telugu, Urdu-origin) and both genders
to ensure the model is exposed to diverse phonological patterns.

---

## TASK-1: Model Implementation

All three models are implemented from scratch in `models.py` using PyTorch primitives.
They share a common character-level language modelling formulation:

```
Input  : <SOS> c₁ c₂ … cₙ
Target : c₁    c₂ c₃ … cₙ <EOS>
```

At each position the model predicts the next character given all preceding characters.
The vocabulary consists of 3 special tokens (`<PAD>`, `<SOS>`, `<EOS>`) plus all unique
characters in the training set (~58 lowercase and uppercase Latin letters).

---

### Model 1 — Vanilla RNN

#### Architecture

```
Input tokens  (batch × seq_len)
      ↓
Embedding    (vocab_size → 64)
      ↓  [dropout 0.3]
nn.RNN       (64 → 256,  2 layers,  dropout=0.3 between layers)
      ↓  [dropout 0.3]
Linear       (256 → vocab_size)
      ↓
Logits       (batch × seq_len × vocab_size)
```

#### Description

A standard Elman RNN with additive recurrence:

```
hₜ = tanh( Wₓ·xₜ  +  Wₕ·hₜ₋₁  +  b )
```

Strengths: minimal parameter count, very fast training.  
Weaknesses: known vanishing-gradient problems for sequences longer than ~10 tokens;
struggles to model long-range phonological dependencies.

#### Hyperparameters

| Hyperparameter | Value |
|---|---|
| `embed_dim` | 64 |
| `hidden_size` | 256 |
| `num_layers` | 2 |
| `dropout` | 0.3 |
| Learning rate | 1 × 10⁻³ (Adam) |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Batch size | 32 |
| Epochs | 100 (early stopping, patience=15) |

#### Trainable Parameters (vocab_size ≈ 80)

| Component | Shape | Parameters |
|---|---|---|
| Embedding | 80 × 64 | 5 120 |
| RNN Layer 0 | (64+256) × 256 × 3 | ~82 432 |
| RNN Layer 1 | (256+256) × 256 × 3 | ~131 584 |
| Linear | 256 × 80 | 20 480 |
| **Total** | | **≈ 239 616** |

---

### Model 2 — Bidirectional LSTM (BLSTM)

#### Architecture

```
Input tokens  (batch × seq_len)
      ↓
Embedding    (vocab_size → 64)
      ↓  [dropout 0.3]
nn.LSTM      (64 → 128,  bidirectional=True,  2 layers)
  → output dimension = 2 × 128 = 256  (forward ‖ backward concatenated)
      ↓  [dropout 0.3]
Linear       (256 → vocab_size)
      ↓
Logits       (batch × seq_len × vocab_size)
```

#### Description

Each LSTM cell maintains both a hidden state *h* and a cell state *c*:

```
fₜ = σ( Wf·[hₜ₋₁, xₜ] + bf )      # forget gate
iₜ = σ( Wi·[hₜ₋₁, xₜ] + bi )      # input gate
c̃ₜ = tanh( Wc·[hₜ₋₁, xₜ] + bc )   # candidate cell
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ          # cell update
oₜ = σ( Wo·[hₜ₋₁, xₜ] + bo )      # output gate
hₜ = oₜ ⊙ tanh(cₜ)
```

The bidirectional wrapper processes the sequence left-to-right *and* right-to-left
simultaneously during **training**, giving each position access to future context.
This helps the model learn better internal representations (e.g., recognising common
Indian name suffixes like *-esh*, *-inder*, *-priya*).  

**Inference note:** A BiLSTM cannot generate token-by-token because it needs the full
sequence for the backward pass. The forward half of the BiLSTM is used autoregressively
at generation time — the model was still *trained* bidirectionally, giving it richer
gradient signal and stronger embeddings.

#### Hyperparameters

| Hyperparameter | Value |
|---|---|
| `embed_dim` | 64 |
| `hidden_size` | 128 (per direction; effective = 256) |
| `num_layers` | 2 |
| `dropout` | 0.3 |
| Learning rate | 1 × 10⁻³ |
| Batch size | 32 |
| Epochs | 100 |

#### Trainable Parameters

| Component | Shape | Parameters |
|---|---|---|
| Embedding | 80 × 64 | 5 120 |
| BiLSTM Layer 0 (2 dirs × 4 gates) | ~(64+128)×128×4×2 | ~197 632 |
| BiLSTM Layer 1 (2 dirs × 4 gates) | ~(256+128)×128×4×2 | ~393 216 |
| Linear | 256 × 80 | 20 480 |
| **Total** | | **≈ 616 448** |

---

### Model 3 — RNN with Basic Attention (AttentionRNN)

#### Architecture

```
Input tokens  (batch × seq_len)
      ↓
Embedding    (vocab_size → 64)
      ↓  [dropout 0.3]
nn.RNN       (64 → 256,  2 layers)
      ↓  rnn_output: (batch × seq_len × 256)
Additive Attention
  score(hₜ) = vᵀ · tanh(W · hₜ)
  α          = softmax(scores)          # (batch × seq_len)
  context    = Σₜ αₜ · hₜ             # (batch × 256)
      ↓  broadcast context → (batch × seq_len × 256)
Concatenate [rnn_output ‖ context]     # (batch × seq_len × 512)
      ↓  [dropout 0.3]
Linear       (512 → vocab_size)
      ↓
Logits       (batch × seq_len × vocab_size)
```

#### Description

A plain RNN is augmented with a Bahdanau-style additive self-attention module.
After the RNN produces hidden states for all positions, a single *global* context
vector is computed as a weighted sum of those states:

```
energy   = tanh( W · H )          W ∈ ℝ^{H×H}
attention = softmax( vᵀ · energy ) v ∈ ℝ^H
context  = Hᵀ · attention
```

This context is then broadcast and concatenated with each position's RNN output
before the final linear projection.  
The attention mechanism gives the model a "holistic" view of the partial sequence
generated so far — it can, for example, avoid repeating a syllable it has already
seen. This is a step beyond pure left-to-right memory carried only through the
hidden state.

#### Hyperparameters

| Hyperparameter | Value |
|---|---|
| `embed_dim` | 64 |
| `hidden_size` | 256 |
| `num_layers` | 2 |
| `dropout` | 0.3 |
| Learning rate | 1 × 10⁻³ |
| Batch size | 32 |
| Epochs | 100 |

#### Trainable Parameters

| Component | Shape | Parameters |
|---|---|---|
| Embedding | 80 × 64 | 5 120 |
| RNN Layer 0 | ~(64+256)×256×3 | ~82 432 |
| RNN Layer 1 | ~(256+256)×256×3 | ~131 584 |
| Attention W | 256 × 256 | 65 792 |
| Attention v | 256 × 1 | 256 |
| Linear | 512 × 80 | 40 960 |
| **Total** | | **≈ 326 144** |

---

### Parameter Count Summary

| Model | Trainable parameters (this codebase, vocab ≈ 55) |
|---|---|
| VanillaRNN | **231 671** |
| BLSTM | **611 575** |
| AttentionRNN | **311 799** |

---

## TASK-2: Quantitative Evaluation

### Metrics

**Novelty Rate**

```
Novelty Rate = (# generated names not in training set) / (# generated names) × 100
```

Higher novelty means the model is generalising to new names rather than memorising.

**Diversity**

```
Diversity = |{unique generated names}| / |generated names|
```

A diversity of 1.0 means every generated name is different; 0.0 means the model
always produces the same name.

### Measured Results (this run)

Figures below match the current `results/evaluation_results.json` after training and
evaluation: **200** names per model, temperature **0.8**, vocab size **55**.

| Model | Novelty Rate | Diversity | Avg Length (chars) | Char coverage |
|---|---|---|---|---|
| VanillaRNN | **92.00%** | **0.975** | 6.37 ± 1.39 | 0.9615 |
| BLSTM | **90.50%** | **0.985** | 6.43 ± 1.57 | 1.0000 |
| AttentionRNN | **92.00%** | **0.990** | 6.86 ± 1.82 | 1.0000 |

*If you re-run `generate_names.py` without fixing the random seed, novelty and
diversity can differ slightly from the table above; re-run `evaluate.py` (or
`main.py` end-to-end) to refresh this JSON.*

### Interpretation

- All three models achieve **strong diversity** (~0.97–0.99) and **high novelty**
  (~90–92%), with name-like lengths (~6–7 characters on average).
- **BLSTM** is trained with **prefix-aligned causal loss** and decoded with **prefix
  re-encoding** (`train.py`, `generate_names.py`).
- **AttentionRNN** now uses **causal attention** (pool only past RNN states per step)
  and **prefix re-encoding** at inference, so training and sampling are aligned.
  Earlier global-attention + single-step decoding versions produced degenerate samples;
  that failure mode is resolved in the current code.
- Temperature controls the creativity–coherence trade-off:
  - T < 0.7 → safer, more repetitive
  - T ≈ 0.8 → default used above
  - T > 1.0 → noisier, more garbled strings

---

## TASK-3: Qualitative Analysis

### Realism of Generated Names

**VanillaRNN** — Generated names tend to be short and phonologically simple.
The model learns common Indian name patterns but can produce odd spellings or clusters.  
Representative samples (`generated/VanillaRNN_generated.txt`, aligned with Task-2 JSON):

```
Suthan, Amutra, Umayal, Anindi, Amisha, Teeta, Amarti, Amaraman, Ramindean, Aljitsan, Arisha, Anprajesh
```

**BLSTM** — After prefix-aligned training, generations resemble plausible Indian-style
given names (compound stems, *-deep* / *-esh* style endings, familiar rhythms).
Representative samples (`generated/BLSTM_generated.txt`):

```
Shureth, Naedeep, Darpan, Sharamti, Sooni, Sumar, Aanya, Prija, Bhavekh, Jetya, Hirad, Anirti
```

**AttentionRNN** — With causal attention and prefix decoding, outputs are now
comparable in length and realism to the other models; occasional odd spellings remain.  
Representative samples (`generated/AttentionRNN_generated.txt`):

```
Ramanira, Onkav, Darvipree, Rajan, Veera, Anna, Hertendra, Amandik, Sunal, Mohif, Vishal, Andha
```

### Common Failure Modes

| Failure Mode | Likely Cause | Affected Models |
|---|---|---|
| **Truncation** — names of 1–2 chars | EOS predicted too early under sampling | All (rare) |
| **Odd spellings / clusters** | Character LM, finite training set | All |
| **Repetition loops** | Older **non-causal** attention + wrong decode (fixed in current code) | AttentionRNN (legacy setup) |
| **Awkward clusters** | Temperature too high | All (T > 1.0) |

### Recommendations

1. **Lower temperature (0.6–0.7)** for more realistic names at the cost of diversity.
2. **Minimum length filtering** — discard generated names shorter than 3 chars.
3. **Beam search** instead of sampling can improve coherence for the RNN models.
4. **Larger dataset** — 5 000+ names would substantially reduce failure modes.

---

## Conclusion

All three models produce plausible, novel Indian-style names under the chosen
hyperparameters. **VanillaRNN** is the smallest and fastest to train; **BLSTM**
adds capacity and slightly different validation dynamics with prefix-aligned training;
**AttentionRNN** matches strong diversity once attention is **causal** and decoding
uses **prefix re-encoding**.

The character-level approach is well-suited to Indian names because it generalises
across the many morphological patterns present in Hindi, Tamil, Bengali, Punjabi, and
other languages without requiring language-specific preprocessing.
