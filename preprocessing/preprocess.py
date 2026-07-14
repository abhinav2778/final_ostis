# ─────────────────────────────────────────────────────────────────
# OSTIS — Module 2: Preprocessing
# Input  : storage/articles.jsonl
# Output : storage/processed_articles.csv
#
# Design notes:
#   - CVE IDs and public IP addresses are extracted BEFORE the text
#     is stripped of punctuation, then re-injected at the end. This
#     avoids the classic bug where cleaning breaks "CVE-2024-1234"
#     into "CVE 2024 1234" or worse, drops it entirely.
#   - Private/reserved IP ranges are filtered out at extraction time
#     since they're not meaningful threat indicators.
#   - Quality filters remove short/near-empty articles, duplicates,
#     and articles with junk titles (navigation pages, etc.)
# ─────────────────────────────────────────────────────────────────

import json
import os
import re
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH  = os.path.join(BASE_DIR, "storage", "articles.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "storage", "processed_articles.csv")

STOP_WORDS = set(stopwords.words("english"))

# ── Private/reserved IP prefixes to exclude ────────────────────────
PRIVATE_PREFIXES = (
    "127.", "192.168.", "10.", "0.", "255.",
    "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.",
    "172.28.", "172.29.", "172.30.", "172.31.",
)

# Strict octet-range IPv4 regex — avoids matching things like version
# numbers or arbitrary dotted numbers that aren't real IPs.
IP_REGEX = re.compile(
    r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
)

CVE_REGEX = re.compile(r'CVE-\d{4}-\d{4,7}', re.IGNORECASE)


def is_private_ip(ip: str) -> bool:
    return ip.startswith(PRIVATE_PREFIXES)


def clean_article_text(raw_text: str) -> str:
    """
    Clean a single article's text while preserving CVE IDs and public
    IP addresses through the cleaning process.
    """
    text = raw_text or ""

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Step 1: extract indicators BEFORE any character stripping
    cve_ids = sorted(set(c.upper() for c in CVE_REGEX.findall(text)))
    ip_addresses = sorted(set(
        ip for ip in IP_REGEX.findall(text) if not is_private_ip(ip)
    ))

    # Step 2: strip everything except letters, numbers, basic punctuation
    text = re.sub(r"[^a-zA-Z0-9.,\- ]", " ", text)

    # Step 3: re-inject preserved indicators
    for cve in cve_ids:
        text += " " + cve
    for ip in ip_addresses:
        text += " " + ip

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Sentence tokenize, then word tokenize + stopword removal per sentence
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    for sent in sentences:
        tokens = word_tokenize(sent.lower())
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
        cleaned_sentences.append(" ".join(tokens))

    return " ".join(cleaned_sentences)


def run():
    print("=" * 60)
    print("OSTIS — Module 2: Preprocessing")
    print("=" * 60)
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}\n")

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: input file not found: {INPUT_PATH}")
        print("Run the crawler first (Module 1).")
        return

    records = []
    skipped_bad_json = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                article = json.loads(line)
            except json.JSONDecodeError:
                skipped_bad_json += 1
                continue

            title = article.get("title", "")
            url   = article.get("url", "")
            text  = article.get("content", "")

            clean_text = clean_article_text(text)

            records.append({
                "title":      title,
                "url":        url,
                "clean_text": clean_text,
            })

    print(f"Raw records loaded : {len(records)}")
    if skipped_bad_json:
        print(f"Skipped malformed JSON lines: {skipped_bad_json}")

    df = pd.DataFrame(records)

    # ── Quality Filters ────────────────────────────────────────────
    before = len(df)
    df = df[df["clean_text"].str.split().str.len() > 100]
    print(f"After min-length filter (>100 words): {len(df)} (removed {before - len(df)})")

    before = len(df)
    df = df.drop_duplicates(subset=["clean_text"])
    print(f"After duplicate removal: {len(df)} (removed {before - len(df)})")

    before = len(df)
    df = df[df["title"].str.strip().str.len() > 10]
    df = df[df["title"] != "nan"]
    print(f"After junk-title filter: {len(df)} (removed {before - len(df)})")

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nPreprocessing complete. Saved {len(df)} records to {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    run()
