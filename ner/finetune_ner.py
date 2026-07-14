# ─────────────────────────────────────────────────────────────────
# OSTIS — Module 5b: NER Fine-Tuning via Distant Supervision
#
# Why: keyword lists miss entities not already known (new malware names,
# new threat actor groups). A fine-tuned token classifier can generalize
# to unseen names by learning contextual patterns instead of exact
# string matches.
#
# Approach: distant supervision — auto-label sentences using the existing
# keyword lists (entity_lists.py) to produce BIO tags, no manual
# annotation needed, then fine-tune RoBERTa on those auto-labels.
#
# BIO tags: B-MALWARE / I-MALWARE / B-ACTOR / I-ACTOR / O
#
# Input  : classifier/tagged_articles.csv
# Output : ner/finetuned_ner/  (saved model + tokenizer + label_map.json)
#
# NOTE: this script downloads roberta-base from huggingface.co and is
# meant to be run on a machine with normal internet access. It is not
# required for the pipeline to function — extract_entities.py (Module 5a)
# works standalone. ner_combiner.py (Module 5c) uses this model if
# present and gracefully falls back to keyword-only if not.
# ─────────────────────────────────────────────────────────────────

import os
import re
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from entity_lists import MALWARE_LIST, THREAT_ACTOR_LIST

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV  = os.path.join(BASE_DIR, "classifier", "tagged_articles.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "ner", "finetuned_ner")
MODEL_NAME = "roberta-base"

LABEL_LIST = ["O", "B-MALWARE", "I-MALWARE", "B-ACTOR", "I-ACTOR"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def label_sentence(sentence: str, malware_list, actor_list):
    tokens = sentence.split()
    labels = ["O"] * len(tokens)

    for entity in malware_list:
        entity_tokens = entity.split()
        n = len(entity_tokens)
        for i in range(len(tokens) - n + 1):
            window = " ".join(tokens[i:i + n]).lower()
            if window == entity:
                labels[i] = "B-MALWARE"
                for j in range(1, n):
                    labels[i + j] = "I-MALWARE"

    for entity in actor_list:
        entity_tokens = entity.split()
        n = len(entity_tokens)
        for i in range(len(tokens) - n + 1):
            window = " ".join(tokens[i:i + n]).lower()
            if window == entity:
                labels[i] = "B-ACTOR"
                for j in range(1, n):
                    labels[i + j] = "I-ACTOR"

    return tokens, labels


class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        tokens_list = [d["tokens"] for d in data]
        labels_list = [d["labels"] for d in data]
        self.encodings = self._tokenize_and_align(tokens_list, labels_list, tokenizer, max_length)

    @staticmethod
    def _tokenize_and_align(tokens_list, labels_list, tokenizer, max_length):
        tokenized = tokenizer(
            tokens_list, is_split_into_words=True, truncation=True,
            max_length=max_length, padding="max_length",
        )
        aligned_labels = []
        for i, label_seq in enumerate(labels_list):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    label_ids.append(LABEL2ID[label_seq[word_id]])
                else:
                    label_ids.append(-100)
                prev_word_id = word_id
            aligned_labels.append(label_ids)
        tokenized["labels"] = aligned_labels
        return tokenized

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.encodings["labels"][idx]),
        }


def build_training_data():
    df = pd.read_csv(INPUT_CSV)
    df = df[df["clean_text"].notna()].reset_index(drop=True)
    print(f"Articles loaded: {len(df)}")

    sentences = []
    for text in df["clean_text"]:
        for part in re.split(r"[.\n]", str(text)):
            words = part.strip().split()
            if 8 <= len(words) <= 60:
                sentences.append(part.strip())
    print(f"Sentences extracted: {len(sentences)}")

    labeled = []
    for sent in sentences:
        tokens, labels = label_sentence(sent, MALWARE_LIST, THREAT_ACTOR_LIST)
        labeled.append({"tokens": tokens, "labels": labels})

    entity_sentences = [s for s in labeled if any(l != "O" for l in s["labels"])]
    print(f"Sentences with entities: {len(entity_sentences)}")

    o_only = [s for s in labeled if all(l == "O" for l in s["labels"])]
    o_sample = random.sample(o_only, min(len(entity_sentences), len(o_only), 2000))

    final_data = entity_sentences + o_sample
    random.shuffle(final_data)
    print(f"Final training samples: {len(final_data)}")
    return final_data


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[ID2LABEL[l] for l in label if l != -100] for label in labels]
    true_preds = [
        [ID2LABEL[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    correct = sum(p == l for preds, lbls in zip(true_preds, true_labels) for p, l in zip(preds, lbls))
    total = sum(len(l) for l in true_labels)
    return {"token_accuracy": round(correct / total, 4) if total else 0.0}


def run(epochs: int = 3, batch_size: int = 16):
    device = get_device()
    print(f"Device: {device}")

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found. Run classification first (Module 3).")
        return

    final_data = build_training_data()
    split = int(0.9 * len(final_data))
    train_data, val_data = final_data[:split], final_data[split:]
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    train_dataset = NERDataset(train_data, tokenizer)
    val_dataset = NERDataset(val_data, tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("=" * 60)
    print("Training NER model via distant supervision...")
    print("=" * 60)
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f)

    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
