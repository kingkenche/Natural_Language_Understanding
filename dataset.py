"""
dataset.py
----------
Dataset and vocabulary utilities for character-level name generation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_IDX   = 0


class NameDataset(Dataset):
    """
    Loads a plain-text file of names (one per line) and exposes them as
    (input_sequence, target_sequence) pairs for next-character prediction.

    Encoding scheme
    ---------------
        input  : <SOS> c1 c2 ... cN
        target : c1    c2 c3 ... cN <EOS>
    """

    def __init__(self, filepath: str):
        self.names            = self._load_names(filepath)
        self.vocab, \
        self.char2idx, \
        self.idx2char         = self._build_vocab()
        self.vocab_size       = len(self.vocab)
        self.max_name_len     = max(len(n) for n in self.names)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_names(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as fh:
            names = [line.strip() for line in fh if line.strip()]
        return names

    def _build_vocab(self):
        chars = set()
        for name in self.names:
            chars.update(name)
        # Fixed ordering: special tokens first, then sorted characters
        vocab     = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + sorted(chars)
        char2idx  = {c: i for i, c in enumerate(vocab)}
        idx2char  = {i: c for c, i in char2idx.items()}
        return vocab, char2idx, idx2char

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, name: str):
        """Return a list of token indices with SOS prefix and EOS suffix."""
        return (
            [self.char2idx[SOS_TOKEN]]
            + [self.char2idx[ch] for ch in name]
            + [self.char2idx[EOS_TOKEN]]
        )

    def decode(self, indices) -> str:
        """Convert a list / tensor of indices back to a name string."""
        chars = []
        for idx in indices:
            idx = int(idx)
            ch  = self.idx2char[idx]
            if ch == EOS_TOKEN:
                break
            if ch not in (PAD_TOKEN, SOS_TOKEN):
                chars.append(ch)
        return "".join(chars)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        encoded = self.encode(self.names[idx])
        x = torch.tensor(encoded[:-1], dtype=torch.long)   # input  : SOS … last_char
        y = torch.tensor(encoded[1:],  dtype=torch.long)   # target : first_char … EOS
        return x, y


# ----------------------------------------------------------------------
# Collate function (pads variable-length sequences in a batch)
# ----------------------------------------------------------------------

def collate_fn(batch):
    """
    Pads sequences to the length of the longest one in the batch.

    Returns
    -------
    x_padded : LongTensor (batch, max_len)
    y_padded : LongTensor (batch, max_len)
    lengths  : LongTensor (batch,)  — original (unpadded) lengths
    """
    xs, ys   = zip(*batch)
    lengths  = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    x_padded = pad_sequence(xs, batch_first=True, padding_value=PAD_IDX)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=PAD_IDX)
    return x_padded, y_padded, lengths


# ----------------------------------------------------------------------
# Quick sanity-check
# ----------------------------------------------------------------------

if __name__ == "__main__":
    ds = NameDataset("TrainingNames.txt")
    print(f"Names       : {len(ds)}")
    print(f"Vocab size  : {ds.vocab_size}")
    print(f"Vocabulary  : {ds.vocab}")
    x, y = ds[0]
    print(f"Sample name : {ds.names[0]}")
    print(f"  x indices : {x.tolist()}")
    print(f"  y indices : {y.tolist()}")
    print(f"  decoded x : {ds.decode(x)}")
