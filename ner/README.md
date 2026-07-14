# Module 5 — Hybrid Named Entity Recognition (NER)

Extracts CVE IDs, IP addresses, malware names, and threat actors from
classified articles.

## Files

- `entity_lists.py` — single source of truth for `MALWARE_LIST`,
  `THREAT_ACTOR_LIST`, and `VENDOR_BLOCKLIST`. Every other NER script
  imports from here — no duplicated lists to fall out of sync.
- `extract_entities.py` — **baseline extractor**. Regex for CVE/IP
  (100% reliable, fixed format), keyword match for malware/actors. No
  model download required — this alone is enough to run the pipeline.
- `finetune_ner.py` — fine-tunes a RoBERTa token classifier via **distant
  supervision**: sentences are auto-labeled using `entity_lists.py`
  (no manual annotation needed), then the model learns to generalize to
  entity names not in the keyword lists. Run this on a machine with
  normal internet access (downloads `roberta-base`).
- `ner_combiner.py` — **production entry point**. Merges keyword output
  with the fine-tuned model's output (if `finetuned_ner/` exists),
  applies `VENDOR_BLOCKLIST` to strip false positives (Microsoft, Cisco,
  AWS, etc. that a general model tends to misclassify as threats), and
  writes `ner_output.csv`. If no fine-tuned model is present, it prints a
  notice and falls back to keyword-only — never crashes.

## Run

```bash
# Baseline only (no model needed):
python ner/extract_entities.py

# Optional: train the model layer (run once, needs internet):
python ner/finetune_ner.py

# Production (uses model if available, else keyword-only):
python ner/ner_combiner.py
```

Input: `classifier/tagged_articles.csv`
Output: `ner/ner_output.csv`

## Why hybrid instead of a single NER model

No public, ungated cybersecurity-specific NER model exists off the shelf.
General-purpose NER models misclassify malware names as generic MISC
entities and often break structured patterns like CVE IDs mid-token.
Regex handles the fixed-format cases with zero ambiguity; the keyword
list handles known entities with zero false positives; the fine-tuned
model catches genuinely new/unseen entity names that neither of the
other two layers can.
