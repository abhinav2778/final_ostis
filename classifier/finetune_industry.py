# ─────────────────────────────────────────────────────────────────
# OSTIS — Module 3c: CySecBERT Fine-tuning — Industry Classification
#
# Task  : classify threat articles into 7 industry domains
# Model : markusbayer/CySecBERT
# Labels: from classifier/tagged_articles.csv (BART bootstrap output)
#
# Class-weighted loss is applied from the start. In every prior version
# of this project, Finance/Healthcare/Education ended up as minority
# classes (14-22 samples vs 400+ for ICS) and the model learned to
# ignore them entirely until class weighting was added. Building it in
# from day one avoids re-discovering that bug.
# ─────────────────────────────────────────────────────────────────

import os
import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGGED_CSV = os.path.join(BASE_DIR, "classifier", "tagged_articles.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "classifier", "finetuned", "industry")
MODEL_NAME = "markusbayer/CySecBERT"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_weighted_trainer_class(class_weights_tensor):
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = CrossEntropyLoss(weight=class_weights_tensor)
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss
    return WeightedTrainer


def run(epochs: int = 6, batch_size: int = 8, max_length: int = 256):
    device = get_device()
    print(f"Device: {device}")

    if not os.path.exists(TAGGED_CSV):
        print(f"ERROR: {TAGGED_CSV} not found.")
        print("Run classify_bootstrap.py first (Module 3b).")
        return

    df = pd.read_csv(TAGGED_CSV)
    df = df[["clean_text", "primary_industry"]].dropna()
    df = df[df["clean_text"].str.split().str.len() > 30]
    df = df.rename(columns={"clean_text": "text", "primary_industry": "industry"})

    print(f"Total samples: {len(df)}")
    print("Industry distribution:")
    print(df["industry"].value_counts().to_string())

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["industry"])
    label_map = {int(i): label for i, label in enumerate(le.classes_)}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    num_labels = len(le.classes_)
    print(f"\nLabel mapping: {label_map}")
    print(f"Number of classes: {num_labels}\n")

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, ignore_mismatched_sizes=True
    ).to(device)

    # ── Class weights — critical for minority-class industries ────
    classes = np.array(sorted(df["label"].unique()))
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=df["label"].values)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {dict(zip([label_map[int(i)] for i in classes], class_weights.round(2)))}\n")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    dataset = Dataset.from_pandas(df[["text", "label"]])
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
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        fp16=False,
        report_to="none",
    )

    WeightedTrainer = make_weighted_trainer_class(class_weights_tensor)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )

    print("=" * 60)
    print("Training industry classifier (class-weighted)...")
    print("=" * 60)
    trainer.train()

    results = trainer.evaluate()
    print("Final evaluation:", results)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Label map saved to: {OUTPUT_DIR}/label_map.json")


if __name__ == "__main__":
    run()
