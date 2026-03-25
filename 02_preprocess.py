"""
02_preprocess.py
─────────────────────────────────────────────────────────────────────────────
Preprocessing pipeline for the IIT Jodhpur corpus.

Steps performed (as required by the assignment):
  1. Load raw documents (scraped files OR built-in synthetic corpus)
  2. Remove boilerplate artefacts  (HTML tags, URLs, emails, …)
  3. Lowercase
  4. Remove excessive punctuation and non-textual content
  5. Tokenise into words
  6. Remove stopwords (built-in list, no NLTK download needed)
  7. Save cleaned corpus  →  corpus/clean_corpus.txt
  8. Report dataset statistics  (documents, tokens, vocab)
  9. Generate Word Cloud image  →  outputs/wordcloud.png

Usage:
    python 02_preprocess.py
─────────────────────────────────────────────────────────────────────────────
"""

import re, os, sys, json, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ── Paths ─────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent
CLEAN_FILE = BASE / "corpus" / "clean_corpus.txt"
SENT_FILE  = BASE / "corpus" / "sentences.json"
OUT_DIR    = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)
(BASE / "corpus").mkdir(exist_ok=True)

# ── English stop-words (built-in – no NLTK download) ──────────────────────
STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and",
    "any","are","aren","as","at","be","because","been","before","being",
    "below","between","both","but","by","can","cannot","could","couldn",
    "did","didn","do","does","doesn","doing","don","down","during","each",
    "few","for","from","further","get","got","had","hadn","has","hasn",
    "have","haven","having","he","her","here","hers","herself","him",
    "himself","his","how","i","if","in","into","is","isn","it","its",
    "itself","just","ll","me","mightn","more","most","mustn","my",
    "myself","needn","no","nor","not","now","of","off","on","once","only",
    "or","other","our","ours","ourselves","out","over","own","re","s",
    "same","shan","she","should","shouldn","so","some","such","t","than",
    "that","the","their","theirs","them","themselves","then","there",
    "these","they","this","those","through","to","too","under","until",
    "up","us","ve","very","was","wasn","we","were","weren","what","when",
    "where","which","while","who","whom","why","will","with","won","would",
    "wouldn","you","your","yours","yourself","yourselves","also","it","its",
    "etc","may","must","shall","well","one","two","three","four",
    "five","six","seven","eight","nine","ten","per","ii","iii",
    # timetable / formatting noise from IIT Jodhpur site
    "lectures","lecture","tutorial","tutorials","lab","labs",
    "mon","tue","wed","thu","fri","sat","sun",
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
    "lt","th","nd","rd","st","na","nil","hrs","hr","min",
}

# ── Load raw text ─────────────────────────────────────────────────────────
def load_documents() -> list[str]:
    """
    Priority:
      1. Scraped .txt files in corpus/scraped/
      2. Synthetic corpus in corpus/raw_corpus.py
    """
    docs: list[str] = []

def load_documents() -> list[str]:
    """
    Load documents in this priority order:
      1. corpus/scraped/*.txt  (from 01_scrape_iitj_final.py)
      2. corpus/manual/*.html  (manually saved browser pages)
      3. corpus/raw_corpus.py  (built-in synthetic corpus – last resort)

    Searches relative to BOTH the script location AND the current
    working directory so it works no matter where you run from.
    """
    docs: list[str] = []

    # Build candidate base directories to search
    search_bases = [BASE, Path.cwd(), Path.cwd().parent]
    # Also look in sibling dirs named iitj_word2vec*
    for p in Path.cwd().parent.glob("iitj_word2vec*"):
        if p.is_dir():
            search_bases.append(p)

    def find_dir(name: str) -> Path | None:
        for base in search_bases:
            d = base / name
            if d.exists():
                return d
        return None

    # ── 1. Scraped .txt files ─────────────────────────────────────────
    scraped_dir = find_dir("corpus/scraped")
    if scraped_dir:
        for fp in sorted(scraped_dir.glob("*.txt")):
            text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            if len(text.split()) >= 20:
                docs.append(text)
        if docs:
            print(f"  Loaded {len(docs)} scraped documents from {scraped_dir}")

    # ── 2. Manually saved HTML files ──────────────────────────────────
    manual_dir = find_dir("corpus/manual")
    if manual_dir:
        from bs4 import BeautifulSoup
        for fp in sorted(manual_dir.glob("*.htm*")):
            html = fp.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script","style","nav","header","footer"]):
                tag.decompose()
            text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
            if len(text.split()) >= 20:
                docs.append(text)
        if docs:
            print(f"  + Also loaded manual HTML from {manual_dir}")

    if docs:
        return docs

    # ── 3. Synthetic fallback (raw_corpus.py) ─────────────────────────
    # Search for raw_corpus.py in all candidate locations
    for base in search_bases:
        rc = base / "corpus" / "raw_corpus.py"
        if rc.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("raw_corpus", rc)
            mod  = importlib.util.load_from_spec(spec) if False else None
            # simpler: just exec the file
            ns = {}
            exec(rc.read_text(), ns)
            raw_docs = [d.strip() for d in ns.get("DOCUMENTS", []) if d.strip()]
            if raw_docs:
                print(f"  Loaded {len(raw_docs)} synthetic docs from {rc}")
                print("  (run 01_scrape_iitj_final.py to use real scraped data)")
                return raw_docs

    raise FileNotFoundError(
        "\n\n  ✗  No corpus found!\n"
        "  Make sure the 'corpus/scraped/' folder (from 01_scrape_iitj_final.py)\n"
        "  is inside the same iitj_word2vec folder as this script.\n\n"
        "  Expected layout:\n"
        "    iitj_word2vec/\n"
        "      02_preprocess.py   ← this script\n"
        "      corpus/\n"
        "        scraped/         ← your 160 .txt files go here\n"
    )


# ── Cleaning helpers ───────────────────────────────────────────────────────
def remove_boilerplate(text: str) -> str:
    """Remove URLs, emails, HTML tags, digits-only tokens, repeated chars."""
    text = re.sub(r"https?://\S+", " ", text)           # URLs
    text = re.sub(r"\S+@\S+\.\S+", " ", text)           # emails
    text = re.sub(r"<[^>]+>", " ", text)                 # HTML tags
    text = re.sub(r"\b\d+\b", " ", text)                 # standalone numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)             # keep only letters
    text = re.sub(r"\b(.)\1{3,}\b", " ", text)           # repeated chars
    return text


def tokenise(text: str) -> list[str]:
    """Lowercase, remove boilerplate, split on whitespace."""
    text = remove_boilerplate(text.lower())
    tokens = text.split()
    # Keep only alphabetic tokens of length >= 2; drop stopwords
    tokens = [t for t in tokens
              if t.isalpha() and len(t) >= 2 and t not in STOPWORDS]
    return tokens


# ── Sentence-level tokenisation for Word2Vec ──────────────────────────────
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def to_sentences(text: str) -> list[list[str]]:
    """Split text into sentences, each sentence into tokens."""
    sentences = _SENT_SPLIT.split(text)
    result = []
    for sent in sentences:
        tokens = tokenise(sent)
        if len(tokens) >= 3:           # skip trivially short sentences
            result.append(tokens)
    return result


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  TASK 1 – DATASET PREPARATION")
    print("=" * 60)

    # 1. Load
    print("\n[1/4] Loading documents …")
    documents = load_documents()

    # 2. Preprocess
    print("[2/4] Preprocessing …")
    all_sentences: list[list[str]] = []
    all_tokens:    list[str]       = []
    clean_docs:    list[str]       = []

    for doc in documents:
        sents = to_sentences(doc)
        all_sentences.extend(sents)
        doc_tokens = [t for sent in sents for t in sent]
        all_tokens.extend(doc_tokens)
        clean_docs.append(" ".join(doc_tokens))

    vocab = sorted(set(all_tokens))

    # 3. Save clean corpus
    print("[3/4] Saving clean corpus …")
    CLEAN_FILE.write_text("\n".join(clean_docs), encoding="utf-8")

    with open(SENT_FILE, "w") as f:
        json.dump(all_sentences, f)

    # 4. Statistics
    print("[4/4] Computing statistics …")
    freq = collections.Counter(all_tokens)

    print("\n" + "─" * 50)
    print(f"  Total documents  : {len(documents):>8,}")
    print(f"  Total sentences  : {len(all_sentences):>8,}")
    print(f"  Total tokens     : {len(all_tokens):>8,}")
    print(f"  Vocabulary size  : {len(vocab):>8,}")
    print(f"  Avg tokens/doc   : {len(all_tokens)/len(documents):>8.0f}")
    print(f"  Avg tokens/sent  : {len(all_tokens)/len(all_sentences):>8.1f}")
    print("─" * 50)
    print("\n  Top 30 most frequent words:")
    for word, cnt in freq.most_common(30):
        print(f"    {word:<25} {cnt:>5}")

    # 5. Word Cloud
    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        colormap="viridis",
        max_words=200,
        collocations=False,
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("IIT Jodhpur Corpus – Word Cloud", fontsize=18, pad=16, fontweight="bold")
    fig.tight_layout()
    wc_path = OUT_DIR / "wordcloud.png"
    fig.savefig(wc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Word cloud saved → {wc_path}")

    # Save stats for the report
    stats = {
        "n_documents":  len(documents),
        "n_sentences":  len(all_sentences),
        "n_tokens":     len(all_tokens),
        "vocab_size":   len(vocab),
        "top30": freq.most_common(30),
    }
    with open(BASE / "corpus" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n  Preprocessing complete.\n")
    return stats


if __name__ == "__main__":
    main()
