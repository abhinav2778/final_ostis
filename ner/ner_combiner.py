# ─────────────────────────────────────────────────────────────────
# OSTIS — Module 5c: Hybrid NER Combiner
#
# Merges the always-available keyword/regex baseline (extract_entities.py)
# with a fine-tuned token-classification model (finetune_ner.py output),
# when present. If the fine-tuned model directory doesn't exist, this
# script degrades gracefully to keyword-only output — the pipeline never
# hard-fails just because fine-tuning hasn't been run yet.
#
# Input  : classifier/tagged_articles.csv
# Output : ner/ner_output.csv
# ─────────────────────────────────────────────────────────────────

import os
import re
import pandas as pd

from entity_lists import MALWARE_LIST, THREAT_ACTOR_LIST, VENDOR_BLOCKLIST, AMBIGUOUS_TERM_CONTEXT

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "ner", "finetuned_ner")
INPUT_CSV  = os.path.join(BASE_DIR, "classifier", "tagged_articles.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "ner", "ner_output.csv")

CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
IP_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
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
    return {kw: re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in keyword_list}


_MALWARE_PATTERNS = _compile_boundary_patterns(MALWARE_LIST)
_ACTOR_PATTERNS = _compile_boundary_patterns(THREAT_ACTOR_LIST)
_AMBIGUOUS_CONTEXT_PATTERNS = {
    term: [re.compile(r'\b' + re.escape(ctx) + r'\b', re.IGNORECASE) for ctx in contexts]
    for term, contexts in AMBIGUOUS_TERM_CONTEXT.items()
}

# clean_text has no sentence-boundary punctuation (preprocess.py strips it),
# so a word-proximity window is used instead of "same sentence".
_AMBIGUOUS_WINDOW = 10


def _passes_ambiguous_check(keyword: str, text: str) -> bool:
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


def keyword_extract(text: str):
    malware = set(
        kw for kw, pattern in _MALWARE_PATTERNS.items()
        if pattern.search(text) and _passes_ambiguous_check(kw, text)
    )
    actors = set(kw for kw, pattern in _ACTOR_PATTERNS.items() if pattern.search(text))
    return malware, actors


def combine_and_filter(kw_entities, model_entities):
    combined = set(kw_entities) | set(model_entities)
    return {
        e for e in combined
        if e.lower() not in VENDOR_BLOCKLIST and len(e) > 2 and not e.isdigit()
    }


class ModelNER:
    """Wraps the fine-tuned token classifier. Returns (None, None) for both
    outputs if the model can't be loaded, so callers can detect and skip
    the model layer cleanly."""

    def __init__(self, model_dir):
        self.available = False
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification

            if not os.path.exists(os.path.join(model_dir, "config.json")):
                print(f"No fine-tuned NER model found at {model_dir} — using keyword-only extraction.")
                return

            self.device = torch.device("mps") if torch.backends.mps.is_available() else (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir).to(self.device)
            self.model.eval()
            self.id2label = self.model.config.id2label
            self.torch = torch
            self.available = True
            print(f"Fine-tuned NER model loaded from {model_dir}.")
        except Exception as e:
            print(f"Could not load fine-tuned NER model ({e}) — using keyword-only extraction.")

    def extract(self, text: str):
        if not self.available:
            return set(), set()

        malware, actors = set(), set()
        for sent in re.split(r"[.\n]", text):
            words = sent.strip().split()
            if len(words) < 3:
                continue
            inputs = self.tokenizer(
                words, is_split_into_words=True, truncation=True,
                max_length=128, return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            preds = self.torch.argmax(outputs.logits, dim=2)[0].tolist()
            word_ids = inputs.word_ids(batch_index=0)

            current_entity, current_type = [], None
            for token_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                label = self.id2label[preds[token_idx]]
                if label.startswith("B-"):
                    if current_entity:
                        e = " ".join(current_entity).lower()
                        (malware if current_type == "MALWARE" else actors).add(e)
                    current_entity, current_type = [words[word_id]], label[2:]
                elif label.startswith("I-") and current_type == label[2:]:
                    if word_id < len(words):
                        current_entity.append(words[word_id])
                else:
                    if current_entity:
                        e = " ".join(current_entity).lower()
                        (malware if current_type == "MALWARE" else actors).add(e)
                    current_entity, current_type = [], None
            if current_entity:
                e = " ".join(current_entity).lower()
                (malware if current_type == "MALWARE" else actors).add(e)

        return malware, actors


def run():
    print("=" * 60)
    print("OSTIS — Module 5c: Hybrid NER Combiner")
    print("=" * 60)

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found. Run classification first (Module 3).")
        return

    model_ner = ModelNER(MODEL_DIR)

    df = pd.read_csv(INPUT_CSV).fillna("")
    print(f"Articles: {len(df)}")

    results = []
    for i, row in df.iterrows():
        text = str(row.get("clean_text", ""))

        cves = sorted(set(CVE_PATTERN.findall(text)))
        cves = [c.upper() for c in cves]
        ips = sorted(set(ip for ip in IP_PATTERN.findall(text) if not is_private_ip(ip)))

        kw_malware, kw_actors = keyword_extract(text)
        model_malware, model_actors = model_ner.extract(text)

        malware = sorted(combine_and_filter(kw_malware, model_malware))
        actors = sorted(combine_and_filter(kw_actors, model_actors))

        results.append({
            "title":            row.get("title", ""),
            "url":              row.get("url", ""),
            "primary_industry": row.get("primary_industry", ""),
            "threat_score":     row.get("threat_score", ""),
            "cve_ids":          ", ".join(cves) if cves else "",
            "ip_addresses":     ", ".join(ips) if ips else "",
            "malware_names":    ", ".join(malware) if malware else "",
            "threat_actors":    ", ".join(actors) if actors else "",
            "clean_text":       text,
        })

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(df)}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nArticles with malware : {(out_df['malware_names'] != '').sum()}")
    print(f"Articles with actors  : {(out_df['threat_actors'] != '').sum()}")
    print(f"Articles with CVEs    : {(out_df['cve_ids'] != '').sum()}")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
