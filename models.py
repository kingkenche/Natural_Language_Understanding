"""
models.py
---------
Three character-level sequence models for Indian name generation.

Models implemented
------------------
1. VanillaRNN      – Plain multi-layer RNN
2. BidirectionalLSTM (BLSTM) – Bidirectional LSTM
3. AttentionRNN    – RNN augmented with additive self-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 1. Vanilla RNN
# ======================================================================

class VanillaRNN(nn.Module):
    """
    Architecture
    ------------
    Embedding(vocab_size → embed_dim)
        ↓
    nn.RNN(embed_dim → hidden_size, num_layers layers, dropout)
        ↓
    Linear(hidden_size → vocab_size)

    Parameters (default config, vocab≈80)
    --------------------------------------
    Embedding : vocab × 64          ≈   5 120
    RNN       : layer-0  64→256     ≈  134 144
               layer-1 256→256     ≈  197 120
    Linear    :         256→vocab  ≈  20 736
    Total                          ≈ 357 120
    """

    def __init__(
        self,
        vocab_size : int,
        embed_dim  : int = 64,
        hidden_size: int = 256,
        num_layers : int = 2,
        dropout    : float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(
            embed_dim, hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, vocab_size)

    # ------------------------------------------------------------------

    def forward(self, x, hidden=None):
        """
        x      : (batch, seq_len)  token indices
        hidden : (num_layers, batch, hidden_size) or None

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        hidden : updated hidden state
        """
        emb            = self.dropout(self.embedding(x))       # (B, L, E)
        out, hidden    = self.rnn(emb, hidden)                 # (B, L, H)
        logits         = self.fc(self.dropout(out))            # (B, L, V)
        return logits, hidden

    def init_hidden(self, batch_size: int, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# 2. Bidirectional LSTM
# ======================================================================

class BidirectionalLSTM(nn.Module):
    """
    Architecture
    ------------
    Embedding(vocab_size → embed_dim)
        ↓
    nn.LSTM(embed_dim → hidden_size, bidirectional=True, num_layers)
        ↓  output dim = 2 × hidden_size  (forward ‖ backward concatenated)
    Linear(2 × hidden_size → vocab_size)

    Notes on use for generation
    ---------------------------
    A BiLSTM must not be stepped with a single new token while pretending
    the backward pass saw the full future sequence. Valid autoregressive
    decoding: at step *t* feed the **entire prefix** ``<SOS>, c₁,…,cₜ``
    through the BiLSTM (length *t+1*). No token in that input is "from the
    future" relative to the next prediction.     Generation code in
    ``generate_names.py`` implements this prefix re-encode loop.
    **Training** must match this: see causal prefix loss for BLSTM in ``train.py``
    (not full-sequence teacher forcing, which leaks future tokens).

    Parameters (default config, vocab≈80)
    --------------------------------------
    Embedding  : vocab × 64          ≈   5 120
    BiLSTM L0  : 2×(64→128) gates   ≈ 100 352
    BiLSTM L1  : 2×(256→128) gates  ≈ 263 168
    Linear     :   256→vocab        ≈  20 736
    Total                           ≈ 389 376
    """

    def __init__(
        self,
        vocab_size : int,
        embed_dim  : int = 64,
        hidden_size: int = 128,   # per direction; output = 2 × hidden_size
        num_layers : int = 2,
        dropout    : float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm    = nn.LSTM(
            embed_dim, hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, vocab_size)

    # ------------------------------------------------------------------

    def forward(self, x, hidden=None):
        """
        x      : (batch, seq_len)
        hidden : tuple (h, c) each (2*num_layers, batch, hidden_size) or None
        """
        emb         = self.dropout(self.embedding(x))
        out, hidden = self.bilstm(emb, hidden)      # out: (B, L, 2H)
        logits      = self.fc(self.dropout(out))    # (B, L, V)
        return logits, hidden

    def init_hidden(self, batch_size: int, device):
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# 3. Additive (Bahdanau-style) Self-Attention module
# ======================================================================

class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention over a sequence of hidden states.

    score(h_t) = v^T · tanh(W · h_t)
    α          = softmax(scores)
    context    = Σ_t α_t · h_t
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        """
        hidden_states : (batch, seq_len, hidden_size)

        Returns
        -------
        context  : (batch, hidden_size)   weighted sum
        weights  : (batch, seq_len)       attention distribution
        """
        energy  = torch.tanh(self.W(hidden_states))        # (B, L, H)
        scores  = self.v(energy).squeeze(-1)               # (B, L)
        weights = F.softmax(scores, dim=-1)                # (B, L)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # (B, H)
        return context, weights


# ======================================================================
# 4. RNN with Attention
# ======================================================================

class AttentionRNN(nn.Module):
    """
    Architecture
    ------------
    Embedding(vocab_size → embed_dim)
        ↓
    nn.RNN(embed_dim → hidden_size, num_layers)
        ↓  rnn_output: (B, L, H)
    Per-step AdditiveAttention over **prefix** hidden states only (causal).
        ↓  concat [ h_t ‖ context_t ] → (B, L, 2H)
    Linear(2 × hidden_size → vocab_size)

    At time *t*, context is a weighted sum of ``h_0 … h_t`` only, matching
    autoregressive decoding (no lookahead). Use **prefix re-encoding** in
    ``generate_names.py`` so attention sees the full history each step.

    Parameters (default config, vocab≈80)
    --------------------------------------
    Embedding    : vocab × 64        ≈   5 120
    RNN L0       : 64→256            ≈  82 432
    RNN L1       : 256→256           ≈ 131 584
    Attention W  : 256×256           ≈  65 792
    Attention v  : 256×1             ≈     256
    Linear       : 512→vocab         ≈  41 472
    Total                            ≈ 326 656
    """

    def __init__(
        self,
        vocab_size : int,
        embed_dim  : int = 64,
        hidden_size: int = 256,
        num_layers : int = 2,
        dropout    : float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(
            embed_dim, hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        # concat(rnn_out, context) → 2 × hidden_size
        self.fc        = nn.Linear(hidden_size * 2, vocab_size)

    # ------------------------------------------------------------------

    def forward(self, x, hidden=None):
        """
        x      : (batch, seq_len)
        hidden : (num_layers, batch, hidden_size) or None
        """
        emb             = self.dropout(self.embedding(x))   # (B, L, E)
        rnn_out, hidden = self.rnn(emb, hidden)               # (B, L, H)
        B, L, _         = rnn_out.shape

        # Causal attention: at step t, pool only h[:, :t+1, :].
        logit_parts = []
        for t in range(L):
            prefix   = rnn_out[:, : t + 1, :]                  # (B, t+1, H)
            ctx, _   = self.attention(prefix)                  # (B, H)
            combined = torch.cat([rnn_out[:, t, :], ctx], dim=-1)  # (B, 2H)
            logit_parts.append(self.fc(self.dropout(combined)).unsqueeze(1))
        logits = torch.cat(logit_parts, dim=1)                # (B, L, V)
        return logits, hidden

    def init_hidden(self, batch_size: int, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# Quick sanity-check
# ======================================================================

if __name__ == "__main__":
    V, B, L = 80, 4, 12
    x = torch.randint(0, V, (B, L))

    for cls, kwargs in [
        (VanillaRNN,       dict(vocab_size=V)),
        (BidirectionalLSTM, dict(vocab_size=V)),
        (AttentionRNN,     dict(vocab_size=V)),
    ]:
        model  = cls(**kwargs)
        logits, _ = model(x)
        print(f"{cls.__name__:20s} | params={model.count_parameters():,} | out={tuple(logits.shape)}")
