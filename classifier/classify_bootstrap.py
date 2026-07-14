# ─────────────────────────────────────────────────────────────────
# OSTIS — Module 3b: Zero-Shot Bootstrap Labeling
#
# Why this exists:
#   CySecBERT's industry classifier (finetune_industry.py) needs labeled
#   training data. We don't have hand-labeled data, so we bootstrap it
#   using facebook/bart-large-mnli zero-shot classification. These labels
#   become the pseudo-ground-truth that CySecBERT is fine-tuned on.
#
#   This step only needs to run ONCE to produce the initial training set.
#   After CySecBERT is fine-tuned, production classification uses
#   classify_finetuned.py (much faster, no BART dependency).
#
# Input  : storage/processed_articles.csv
# Output : classifier/tagged_articles.csv
# ─────────────────────────────────────────────────────────────────

import os
import time
import pandas as pd
from transformers import pipeline

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV  = os.path.join(BASE_DIR, "storage", "processed_articles.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "classifier", "tagged_articles.csv")

BINARY_LABELS = [
    "cybersecurity threat malware vulnerability ransomware phishing data "
    "breach botnet hacking fraud espionage zero-day exploit network "
    "intrusion identity theft",
    "general unrelated non-security content",
]

INDUSTRY_LABELS = [
    "Healthcare", "Finance", "Government", "IoT",
    "ICS", "Education", "General Cybersecurity",
]

THREAT_THRESHOLD = 0.55


def run():
    print("=" * 60)
    print("OSTIS — Module 3b: Zero-Shot Bootstrap Labeling (BART)")
    print("=" * 60)
    print("Loading facebook/bart-large-mnli (first run downloads ~1.6GB)...")

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("Model loaded.\n")

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: input file not found: {INPUT_CSV}")
        print("Run preprocessing first (Module 2).")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Total articles loaded: {len(df)}\n")

    results = []
    discarded = 0
    start = time.time()

    for idx, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        url = str(row.get("url", "")).strip()
        text = str(row.get("clean_text", "")).strip()

        classification_input = f"{title}. {text[:300]}"

        binary = classifier(classification_input, candidate_labels=BINARY_LABELS, multi_label=False)
        top_label, top_score = binary["labels"][0], binary["scores"][0]

        if "general unrelated" in top_label or top_score < THREAT_THRESHOLD:
            discarded += 1
            continue

        industry = classifier(classification_input, candidate_labels=INDUSTRY_LABELS, multi_label=True)
        primary_industry, primary_score = industry["labels"][0], industry["scores"][0]

        secondary_industry = ""
        if len(industry["scores"]) > 1 and industry["scores"][1] > 0.40:
            secondary_industry = industry["labels"][1]

        results.append({
            "title": title,
            "url": url,
            "clean_text": text,
            "threat_score": round(top_score, 4),
            "primary_industry": primary_industry,
            "industry_score": round(primary_score, 4),
            "secondary_industry": secondary_industry,
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)}...")

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_CSV, index=False)

    elapsed = round(time.time() - start, 1)
    print(f"\n{'=' * 60}")
    print("BOOTSTRAP LABELING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total processed          : {len(df)}")
    print(f"Threat-relevant (tagged) : {len(results)}")
    print(f"Discarded (not threat)   : {discarded}")
    print(f"Time taken               : {elapsed}s")
    print(f"Output saved to          : {OUTPUT_CSV}")

    if results:
        print("\nIndustry distribution:")
        print(output_df["primary_industry"].value_counts().to_string())


if __name__ == "__main__":
    run()
