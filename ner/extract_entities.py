# ─────────────────────────────────────────────────────────────────
# OSTIS — Module 5a: Baseline Hybrid NER (regex + keyword lists)
#
# This is the always-available extraction layer — no model download
# required. It handles:
#   CVE IDs      → regex (fixed MITRE format, 100% reliable)
#   IP addresses → regex (public IPs only; private ranges filtered)
#   Malware      → keyword list match
#   Threat actor → keyword list match
#
# ner_combiner.py (Module 5b) layers a fine-tuned model on top of this
# to catch entities NOT in the keyword lists — but this script alone is
# enough to run the full pipeline end-to-end.
#
# Input  : classifier/tagged_articles.csv
# Output : ner/ner_output.csv
# ─────────────────────────────────────────────────────────────────

import os
import re
import pandas as pd

from entity_lists import MALWARE_LIST, THREAT_ACTOR_LIST, AMBIGUOUS_TERM_CONTEXT

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV  = os.path.join(BASE_DIR, "classifier", "tagged_articles.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "ner", "ner_output.csv")

CVE_PATTERN = re.compile(r'CVE-\d{4}-\d{4,7}', re.IGNORECASE)
IP_PATTERN = re.compile(
    r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
    r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
)
PRIVATE_PREFIXES = (
    "127.", "192.168.", "10.", "0.", "255.",
    "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.",
    "172.28.", "172.29.", "172.30.", "172.31.",
)


def is_private_ip(ip: str) -> bool:
    return ip.startswith(PRIVATE_PREFIXES)


def _compile_boundary_patterns(keyword_list):
    """Compile word-boundary regex for each keyword so 'conti' doesn't match
    inside 'continuously', 'play' doesn't match inside 'display', etc."""
    return {kw: re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in keyword_list}


_MALWARE_PATTERNS = _compile_boundary_patterns(MALWARE_LIST)
_ACTOR_PATTERNS = _compile_boundary_patterns(THREAT_ACTOR_LIST)
_AMBIGUOUS_CONTEXT_PATTERNS = {
    term: [re.compile(r'\b' + re.escape(ctx) + r'\b', re.IGNORECASE) for ctx in contexts]
    for term, contexts in AMBIGUOUS_TERM_CONTEXT.items()
}

# NOTE: preprocess.py strips sentence-boundary punctuation entirely (periods
# are tokenized then dropped by the len>2 stopword filter), so clean_text
# arrives as one continuous run of words with no periods. Splitting on '.'
# to approximate "same sentence" is therefore a no-op on real data -- it
# silently degrades to "appears anywhere in the whole article", which is
# true for nearly every article that mentions ransomware at all. A word-
# proximity window on the token stream is what actually works here.
_AMBIGUOUS_WINDOW = 10


def _passes_ambiguous_check(keyword: str, text: str) -> bool:
    """For ambiguous keywords (also common English words), only accept a
    match if a context word appears within a small token window of the
    keyword's actual position in the text."""
    if keyword not in AMBIGUOUS_TERM_CONTEXT:
        return True
    tokens = text.lower().split()
    positions = [i for i, t in enumerate(tokens) if t == keyword]
    context_patterns = _AMBIGUOUS_CONTEXT_PATTERNS[keyword]
    for pos in positions:
        start, end = max(0, pos - _AMBIGUOUS_WINDOW), pos + _AMBIGUOUS_WINDOW + 1
        nearby_text = " ".join(tokens[start:end])
        if any(ctx.search(nearby_text) for ctx in context_patterns):
            return True
    return False


def extract_entities(text: str):
    if not isinstance(text, str) or not text.strip():
        return [], [], [], []

    cves = sorted(set(c.upper() for c in CVE_PATTERN.findall(text)))
    ips = sorted(set(ip for ip in IP_PATTERN.findall(text) if not is_private_ip(ip)))

    malware = sorted(
        kw for kw, pattern in _MALWARE_PATTERNS.items()
        if pattern.search(text) and _passes_ambiguous_check(kw, text)
    )
    actors = sorted(kw for kw, pattern in _ACTOR_PATTERNS.items() if pattern.search(text))

    return cves, ips, malware, actors


def run():
    print("=" * 60)
    print("OSTIS — Module 5a: Baseline Hybrid NER (regex + keywords)")
    print("=" * 60)

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found. Run classification first (Module 3).")
        return

    df = pd.read_csv(INPUT_CSV).dropna(subset=["clean_text"])
    print(f"Articles loaded: {len(df)}\n")

    results = []
    totals = {"cve": 0, "ip": 0, "malware": 0, "actor": 0}

    for _, row in df.iterrows():
        cves, ips, malware, actors = extract_entities(row["clean_text"])
        totals["cve"] += len(cves)
        totals["ip"] += len(ips)
        totals["malware"] += len(malware)
        totals["actor"] += len(actors)

        results.append({
            "title":            row.get("title", ""),
            "url":              row.get("url", ""),
            "primary_industry": row.get("primary_industry", ""),
            "threat_score":     row.get("threat_score", ""),
            "cve_ids":          ", ".join(cves) if cves else "",
            "ip_addresses":     ", ".join(ips) if ips else "",
            "malware_names":    ", ".join(malware) if malware else "",
            "threat_actors":    ", ".join(actors) if actors else "",
            "clean_text":       row.get("clean_text", ""),
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_CSV, index=False)

    with_entities = output_df[
        (output_df["cve_ids"] != "") | (output_df["ip_addresses"] != "") |
        (output_df["malware_names"] != "") | (output_df["threat_actors"] != "")
    ].shape[0]

    print(f"CVE IDs found       : {totals['cve']}")
    print(f"IP addresses found  : {totals['ip']}")
    print(f"Malware names found : {totals['malware']}")
    print(f"Threat actors found : {totals['actor']}")
    print(f"Articles w/ entities: {with_entities}/{len(df)}")
    print(f"Output saved to     : {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
