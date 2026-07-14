# ─────────────────────────────────────────────────────────────────
# OSTIS — Module 3a: CySecBERT Fine-tuning — Binary Classification
#
# Task  : classify text as threat (1) or not-threat (0)
# Model : markusbayer/CySecBERT
#
# Positive class : articles from storage/processed_articles.csv
#                  (assumed threat-relevant, since they came from
#                  cybersecurity sources)
# Negative class : AG News (general, non-cybersecurity news),
#                  sampled to match positive class size for balance
# ─────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV  = os.path.join(BASE_DIR, "storage", "processed_articles.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "classifier", "finetuned", "binary")
MODEL_NAME = "markusbayer/CySecBERT"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataset():
    df = pd.read_csv(INPUT_CSV)
    df = df[["clean_text"]].dropna()
    df = df[df["clean_text"].str.split().str.len() > 30]
    df["label"] = 1
    df = df.rename(columns={"clean_text": "text"})

    print(f"Threat (positive) samples: {len(df)}")

    ag_news = load_dataset("ag_news", split="train")
    ag_df = pd.DataFrame(ag_news)[["text"]]
    ag_df["label"] = 0
    ag_sample = ag_df.sample(n=min(len(df), len(ag_df)), random_state=42)

    print(f"Non-threat (negative) samples: {len(ag_sample)}")

    combined = pd.concat([df, ag_sample], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined


def run(epochs: int = 3, batch_size: int = 8, max_length: int = 256):
    device = get_device()
    print(f"Device: {device}")

    print(f"Loading tokenizer/model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    combined = build_dataset()

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    dataset = Dataset.from_pandas(combined[["text", "label"]])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_data = dataset["train"].map(tokenize, batched=True)
    eval_data = dataset["test"].map(tokenize, batched=True)
    train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=50,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )

    print("=" * 60)
    print("Training binary classifier...")
    print("=" * 60)
    trainer.train()

    results = trainer.evaluate()
    print("Final evaluation:", results)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
