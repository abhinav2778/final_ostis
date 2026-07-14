# ─────────────────────────────────────────────────────────────────
# OSTIS — Module 3d: Production Classification (Fine-tuned CySecBERT)
#
# Uses the two fine-tuned models from finetune_binary.py and
# finetune_industry.py — no BART dependency, single forward pass
# per stage, fast enough for the incremental pipeline.
#
# Input  : storage/processed_articles.csv
# Output : classifier/tagged_articles.csv
# ─────────────────────────────────────────────────────────────────

import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV    = os.path.join(BASE_DIR, "storage", "processed_articles.csv")
OUTPUT_CSV   = os.path.join(BASE_DIR, "classifier", "tagged_articles.csv")
BINARY_DIR   = os.path.join(BASE_DIR, "classifier", "finetuned", "binary")
INDUSTRY_DIR = os.path.join(BASE_DIR, "classifier", "finetuned", "industry")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_classifier(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model


def predict(text, tokenizer, model, device, max_length=256):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_class = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return pred_class, confidence


def run():
    device = get_device()
    print(f"Device: {device}")

    for path in (BINARY_DIR, INDUSTRY_DIR):
        if not os.path.exists(path):
            print(f"ERROR: fine-tuned model not found at {path}")
            print("Run finetune_binary.py and finetune_industry.py first.")
            return

    print("Loading binary classifier...")
    binary_tok, binary_model = load_classifier(BINARY_DIR, device)

    print("Loading industry classifier...")
    industry_tok, industry_model = load_classifier(INDUSTRY_DIR, device)

    with open(os.path.join(INDUSTRY_DIR, "label_map.json")) as f:
        label_map = json.load(f)
    print(f"Industry labels: {label_map}\n")

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found. Run preprocessing first.")
        return

    df = pd.read_csv(INPUT_CSV).dropna(subset=["clean_text"])
    print(f"Total articles: {len(df)}\n")

    results = []
    discarded = 0

    for idx, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        url = str(row.get("url", "")).strip()
        text = str(row.get("clean_text", "")).strip()
        input_text = f"{title}. {text[:300]}"

        binary_pred, binary_conf = predict(input_text, binary_tok, binary_model, device)
        if binary_pred == 0:
            discarded += 1
            continue

        industry_pred, industry_conf = predict(input_text, industry_tok, industry_model, device)
        industry_label = label_map[str(industry_pred)]

        results.append({
            "title": title,
            "url": url,
            "clean_text": text,
            "threat_score": round(binary_conf, 4),
            "primary_industry": industry_label,
            "industry_score": round(industry_conf, 4),
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)}...")

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'=' * 60}")
    print("CLASSIFICATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total processed  : {len(df)}")
    print(f"Threats detected : {len(results)}")
    print(f"Discarded        : {discarded}")
    print(f"Output saved to  : {OUTPUT_CSV}")

    if results:
        print("\nIndustry distribution:")
        print(output_df["primary_industry"].value_counts().to_string())


if __name__ == "__main__":
    run()
