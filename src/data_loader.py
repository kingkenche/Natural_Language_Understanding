"""
data_loader.py — Download & preprocess the BBC News dataset (Sport + Politics).

The BBC News dataset is bundled as raw text files organised by category.
We download the public CSV mirror, filter to 'sport' and 'politics',
clean the text, and persist train/test splits.
"""

import os
import re
import string
import urllib.request
import zipfile

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# ── paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_CSV = os.path.join(DATA_DIR, "bbc-news-data.csv")

# ── NLTK resources ───────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ── download ─────────────────────────────────────────────────────────────
def download_dataset():
    """Download the BBC News dataset if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_CSV):
        print(f"[INFO] Dataset already exists at {RAW_CSV}")
        return

    # We use the raw-text zip from the original source
    ZIP_URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
    zip_path = os.path.join(DATA_DIR, "bbc-fulltext.zip")

    print("[INFO] Downloading BBC News dataset …")
    urllib.request.urlretrieve(ZIP_URL, zip_path)

    print("[INFO] Extracting …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)

    # Build a CSV from the extracted text files
    rows = []
    bbc_root = os.path.join(DATA_DIR, "bbc")
    for category in os.listdir(bbc_root):
        cat_dir = os.path.join(bbc_root, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            fpath = os.path.join(cat_dir, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            rows.append({"category": category, "filename": fname, "text": text})

    df = pd.DataFrame(rows)
    df.to_csv(RAW_CSV, index=False)
    print(f"[INFO] Saved {len(df)} documents → {RAW_CSV}")


# ── cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/digits, strip stopwords, lemmatise."""
    text = text.lower()
    text = re.sub(r"\d+", "", text)                     # digits
    text = text.translate(str.maketrans("", "", string.punctuation))  # punct
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


# ── load & split ─────────────────────────────────────────────────────────
def load_data(random_state: int = 42):
    """Return X_train, X_test, y_train, y_test (cleaned text, label)."""
    download_dataset()

    df = pd.read_csv(RAW_CSV)
    df = df[df["category"].isin(["sport", "politics"])].copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df["label"] = df["category"].map({"sport": 1, "politics": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.20, stratify=df["label"], random_state=random_state,
    )
    print(f"[INFO] Train: {len(X_train)}  |  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df = load_data()
    print(df["category"].value_counts())
    print("Sample cleaned text:\n", X_train.iloc[0][:200])
