# Module 2 — Preprocessing

Cleans raw crawled articles and prepares them for classification.

## Run

```bash
python preprocessing/preprocess.py
```

Input: `storage/articles.jsonl`
Output: `storage/processed_articles.csv`

## Design notes

- **CVE/IP preservation**: indicators are extracted with regex *before* the
  text is stripped of punctuation, then re-injected afterward. This avoids
  the common bug where character-stripping mangles `CVE-2024-1234` into
  `CVE 2024 1234` or drops it.
- **Private IP filtering**: only public IPv4 addresses are treated as
  confirmed indicators for re-injection (RFC1918 ranges and loopback are
  excluded). Downstream NER (Module 5) independently re-validates this.
- **Quality filters** applied in order: minimum 100 words after stopword
  removal, drop exact-duplicate clean text, drop junk/empty titles.
