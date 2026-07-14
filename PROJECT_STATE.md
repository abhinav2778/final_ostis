# OSTIS — Project State (read this first in any new chat)

Last updated: Module 1 complete, starting Module 2.

## Why this file exists
This project was rebuilt from scratch after two prior versions (V2, V3) were
lost/corrupted mid-build. This file is the single source of truth for what's
done, what's pending, and key decisions — so a new chat session can resume
instantly by reading this file instead of relying on chat memory.

## Build order & status
- [x] Repo skeleton, .gitignore, requirements.txt, .env.example, README
- [x] Module 1 — Crawler (Scrapy)
- [ ] Module 2 — Preprocessing
- [ ] Module 3 — Classification (CySecBERT fine-tuning: binary + industry)
- [ ] Module 4 — Embeddings + ChromaDB
- [ ] Module 5 — Hybrid NER
- [ ] Module 6 — Knowledge Graph (Neo4j)
- [ ] Module 7 — Incremental Pipeline Runner
- [ ] Module 8 — FastAPI backend
- [ ] Module 9 — Streamlit dashboard
- [ ] Module 10 — Notification engine (email/webhook/SIEM export) — from
      interview feedback, not yet built in any prior version
- [ ] Module 11 — Mobile app (React Native) — consumes same FastAPI
      backend, no new backend logic needed beyond what Module 8 exposes

## Key decisions carried forward from prior versions
- **CySecBERT** over BART: smaller (440MB vs 1.6GB), single forward pass,
  domain-pretrained, offline.
- **Class-weighted loss** required for industry classifier — Finance/
  Healthcare/Education are minority classes; without weighting the model
  ignores them entirely.
- **Hybrid NER**: regex (CVE/IP — 100% reliable, fixed pattern) + keyword
  lists (known malware/actors) + vendor blocklist (prevents Microsoft/
  Cisco/AWS etc. being flagged as threats) + transformer fallback for
  unknown entity discovery.
- **Incremental pipeline**: ledger file (`storage/processed_urls.txt`)
  tracks processed article URLs so re-runs only process the delta, not the
  full corpus.
- **FastAPI over direct DB access from dashboard**: models/DB connections
  loaded once at startup via lifespan events, not per-request.
- **Portable paths only**: every module resolves `BASE_DIR` via
  `os.path.dirname(os.path.abspath(__file__))` — never hardcode absolute
  paths like `/Users/abhinav/...`.
- **Git commit after every module** — this is the fix for what went wrong
  in V2/V3 (long uncommitted stretches, a bad edit wiped days of work).

## Interview requirements (must be included this time)
- SIEM export (CEF format) — was built in a later V2 dashboard iteration,
  needs clean rebuild.
- CVE ID extraction — covered by NER regex.
- Malicious IP extraction — covered by NER regex + IPAddress graph nodes;
  needs private-IP filtering (exclude 10.x, 192.168.x, 127.x, etc.).
- Org notifications (email/webhook) — NOT yet built cleanly in any prior
  version; planned as Module 10.
- CERT-In crawling — built into Module 1 crawler from the start this time.

## Environment
- Python 3.11
- Neo4j Desktop for the knowledge graph (bolt://127.0.0.1:7687)
- Apple M-series MPS backend available for local fine-tuning (adjust
  device selection to `cpu` if running elsewhere)

## What NOT to do
- Don't copy code from OSTIS_V2 files wholesale — that repo was partially
  overwritten during V3 development and is not fully trustworthy. Chat
  history + this file are the source of truth. V2 files may be checked for
  reference only when something here is ambiguous.
