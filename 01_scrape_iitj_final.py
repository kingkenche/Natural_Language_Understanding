"""
01_scrape_iitj_final.py
──────────────────────────────────────────────────────────────────────────
IIT Jodhpur corpus collector — works WITHOUT Selenium.

What this script does:
  1. Fetches homepage → extracts every internal link
  2. Crawls each discovered link (no guessing dead URL patterns)
  3. Downloads every .pdf found and extracts its text
  4. Saves one .txt per page/PDF in  corpus/scraped/

Requirements (all pip-installable on your college machine):
    pip install requests beautifulsoup4 pdfminer.six tqdm

Usage:
    python 01_scrape_iitj_final.py
    # then run:
    python run_all.py
──────────────────────────────────────────────────────────────────────────
"""

import io, os, re, time, json, hashlib, unicodedata, logging
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse, urlencode

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────────────────
BASE          = "https://iitj.ac.in"
BASE_DOMAIN   = "iitj.ac.in"
MIN_WORDS     = 30          # save page if it has at least this many words
MAX_HTML_PAGES= 1500         # cap on HTML pages crawled
MAX_PDF_PAGES = 1000          # cap on PDFs downloaded
CRAWL_DELAY   = 1.0         # seconds between requests (be polite)

OUT_DIR = Path("corpus/scraped")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── English-text filter ───────────────────────────────────────────────────
def is_english(text: str, threshold: float = 0.75) -> bool:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    return sum(1 for c in chars if ord(c) < 128) / len(chars) >= threshold


def clean_soup(soup: BeautifulSoup) -> str:
    """Strip boilerplate tags, return cleaned English text."""
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "iframe", "noscript", "form", "button",
                     "select", "option", "meta", "link"]):
        tag.decompose()
    raw = soup.get_text(separator=" ")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    lines = [ln.strip() for ln in raw.splitlines()
             if is_english(ln.strip()) and len(ln.split()) >= 3]
    return "\n".join(lines)


# ── URL helpers ───────────────────────────────────────────────────────────
def normalise(url: str) -> str:
    p = urlparse(url)
    return urlunparse(p._replace(fragment=""))  # strip #anchor


def is_crawlable(url: str) -> bool:
    p = urlparse(url)
    if BASE_DOMAIN not in p.netloc:
        return False
    if re.search(r"\.(jpg|jpeg|png|gif|svg|ico|css|js|zip|rar|mp4|mp3|woff)$",
                 p.path, re.I):
        return False
    return url.startswith("http")


def is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def url_to_fname(url: str, ext: str = ".txt") -> str:
    h    = hashlib.md5(url.encode()).hexdigest()[:8]
    slug = re.sub(r"[^\w]", "_", urlparse(url).path)[-50:]
    return f"{slug}_{h}{ext}"


# ── HTTP helpers ──────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

def fetch_html(url: str) -> str | None:
    try:
        r = SESSION.get(url, timeout=15)
        r.raise_for_status()
        if "text/html" not in r.headers.get("Content-Type", ""):
            return None
        return r.text
    except Exception as e:
        log.warning(f"  HTML fetch failed {url}: {e}")
        return None

def fetch_bytes(url: str) -> bytes | None:
    try:
        r = SESSION.get(url, timeout=30, stream=True)
        r.raise_for_status()
        return r.content
    except Exception as e:
        log.warning(f"  Bytes fetch failed {url}: {e}")
        return None


# ── PDF extraction ────────────────────────────────────────────────────────
def extract_pdf(data: bytes) -> str | None:
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(io.BytesIO(data))
        lines = [ln.strip() for ln in text.splitlines()
                 if is_english(ln.strip()) and len(ln.split()) >= 3]
        return "\n".join(lines)
    except ImportError:
        log.error("pdfminer.six not installed. Run: pip install pdfminer.six")
        return None
    except Exception as e:
        log.warning(f"  PDF parse error: {e}")
        return None


# ── Link extraction ───────────────────────────────────────────────────────
def extract_links(html: str, base_url: str) -> tuple[list[str], list[str]]:
    """Returns (html_links, pdf_links)."""
    soup  = BeautifulSoup(html, "html.parser")
    html_links, pdf_links = [], []
    for a in soup.find_all("a", href=True):
        raw  = a["href"].strip()
        full = normalise(urljoin(base_url, raw))
        if not is_crawlable(full):
            continue
        if is_pdf_url(full):
            pdf_links.append(full)
        else:
            html_links.append(full)
    return list(set(html_links)), list(set(pdf_links))


# ── Saver ─────────────────────────────────────────────────────────────────
saved:   list[str] = []
visited: set[str]  = set()

def save(text: str, url: str, label: str = "") -> bool:
    words = text.split()
    if len(words) < MIN_WORDS:
        log.warning(f"  Too short ({len(words)} w), skip: {url}")
        return False
    fname = url_to_fname(url)
    (OUT_DIR / fname).write_text(text, encoding="utf-8")
    saved.append(url)
    log.info(f"  ✓ {label}{len(words):,} words → {fname}")
    return True


# ── Phase 1: HTML crawl ───────────────────────────────────────────────────
def crawl_html() -> list[str]:
    """BFS crawl. Returns list of PDF URLs discovered along the way."""
    queue:    list[str] = [BASE + "/"]
    all_pdfs: list[str] = []
    count = 0

    while queue and count < MAX_HTML_PAGES:
        url = queue.pop(0)
        url = normalise(url)
        if url in visited:
            continue
        visited.add(url)
        count += 1

        log.info(f"[HTML {count}/{MAX_HTML_PAGES}] {url}")
        html = fetch_html(url)
        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")
        text = clean_soup(soup)
        save(text, url)

        new_html, new_pdfs = extract_links(html, url)
        all_pdfs.extend(new_pdfs)

        for lnk in new_html:
            if lnk not in visited and lnk not in queue:
                queue.append(lnk)

        time.sleep(CRAWL_DELAY)

    return list(set(all_pdfs))


# ── Phase 2: PDF extraction ───────────────────────────────────────────────
def crawl_pdfs(pdf_urls: list[str]):
    count = 0
    for url in pdf_urls:
        url = normalise(url)
        if url in visited or count >= MAX_PDF_PAGES:
            break
        visited.add(url)
        count += 1

        log.info(f"[PDF {count}/{MAX_PDF_PAGES}] {url}")
        data = fetch_bytes(url)
        if not data:
            continue

        text = extract_pdf(data)
        if text:
            save(text, url, label="[PDF] ")
        time.sleep(CRAWL_DELAY)


# ── Phase 3: Try extra faculty/course pages if we have too few docs ───────
EXTRA_PATTERNS = [
    # Faculty individual pages – common pattern
    "/faculty/",
    "/people/",
    "/courses/",
    "/syllabus/",
    "/curriculum/",
    "/notice/",
    "/announcement/",
    "/newsletter/",
    "/circular/",
    "/placement/",
    "/research/",
]

def try_extra_paths():
    """Try common path patterns that may not be linked from nav."""
    for path in EXTRA_PATTERNS:
        url = BASE + path
        if url in visited:
            continue
        visited.add(url)
        log.info(f"[EXTRA] {url}")
        html = fetch_html(url)
        if not html:
            time.sleep(CRAWL_DELAY)
            continue
        soup = BeautifulSoup(html, "html.parser")
        text = clean_soup(soup)
        save(text, url)
        # Also grab links found here
        new_html, new_pdfs = extract_links(html, url)
        crawl_pdfs(new_pdfs)
        for lnk in new_html[:20]:   # crawl a sample of sublinks
            if lnk in visited:
                continue
            visited.add(lnk)
            h2 = fetch_html(lnk)
            if h2:
                t2 = clean_soup(BeautifulSoup(h2, "html.parser"))
                save(t2, lnk)
                _, p2 = extract_links(h2, lnk)
                crawl_pdfs(p2)
            time.sleep(CRAWL_DELAY)
        time.sleep(CRAWL_DELAY)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  IIT Jodhpur Corpus Collector  (no Selenium needed)")
    log.info("=" * 60)

    log.info("\n── Phase 1: HTML crawl ──────────────────────────────────")
    pdf_urls = crawl_html()

    log.info(f"\n── Phase 2: PDF extraction ({len(pdf_urls)} PDFs found) ──────")
    crawl_pdfs(pdf_urls)

    if len(saved) < 15:
        log.info("\n── Phase 3: Trying extra URL patterns ──────────────────")
        try_extra_paths()

    # ── Report ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(f"  Pages/PDFs visited : {len(visited)}")
    log.info(f"  Files saved        : {len(saved)}")
    log.info(f"  Output directory   : {OUT_DIR}/")
    log.info("=" * 60)

    with open("corpus/scraped_manifest.json", "w") as f:
        json.dump({"saved": saved, "all_visited": list(visited)}, f, indent=2)

    if len(saved) == 0:
        log.error(
            "\n  ✗  Nothing saved. Likely causes:\n"
            "  • Network blocked (check proxy settings)\n"
            "  • All subpages need JavaScript rendering → install Selenium\n"
            "    pip install selenium\n"
            "    https://chromedriver.chromium.org/downloads"
        )
    elif len(saved) < 5:
        log.warning(
            f"\n  ⚠  Only {len(saved)} file(s) saved.\n"
            "  Consider using the manual HTML save method:\n"
            "    1. Open pages in browser, Ctrl+S → 'Web Page, HTML only'\n"
            "    2. Put files in  corpus/manual/\n"
            "    3. Run: python 01_scrape_data_v2.py --manual"
        )
    else:
        log.info(f"\n  ✓  Done! Run next:  python run_all.py")


if __name__ == "__main__":
    main()
