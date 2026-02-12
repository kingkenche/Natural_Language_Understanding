"""
features.py — Vectorisation pipelines: BoW, TF-IDF, Bigram TF-IDF.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_vectorizers():
    """Return a dict of {name: vectorizer} for the three feature methods."""
    return {
        "BoW": CountVectorizer(max_features=5000),
        "TF-IDF": TfidfVectorizer(max_features=5000),
        "Bigram TF-IDF": TfidfVectorizer(ngram_range=(1, 2), max_features=10000),
    }
