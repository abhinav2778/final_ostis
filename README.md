# OSTIS — Organization-Specific Threat Intelligence System

Automated cybersecurity threat intelligence pipeline: crawling → preprocessing →
classification → embeddings → NER → knowledge graph → API → dashboard.

This is a clean rebuild. Every module is committed to git as it's completed so
a bad edit can never wipe out a working state again.

## Build status

- [ ] Module 1 — Crawler (Scrapy)
- [x] Module 2 — Preprocessing
- [ ] Module 3 — Classification (CySecBERT fine-tuning)
- [x] Module 4 — Embeddings + ChromaDB
- [ ] Module 5 — Hybrid NER
- [ ] Module 6 — Knowledge Graph (Neo4j)
- [ ] Module 7 — Incremental Pipeline Runner
- [ ] Module 8 — FastAPI backend
- [ ] Module 9 — Streamlit dashboard
- [ ] Module 10 — Notification engine (email/webhook/SIEM export)
- [ ] Module 11 — Mobile app (React Native)

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your Neo4j password
```

## Folder structure

```
OSTIS/
├── crawler/           # Module 1 — Scrapy spider
├── preprocessing/      # Module 2 — text cleaning
├── classifier/         # Module 3 — CySecBERT classification
├── storage/            # Module 4 — ChromaDB vector store
├── ner/                 # Module 5 — hybrid NER
├── graph/               # Module 6 — Neo4j knowledge graph
├── pipeline/            # Module 7 — incremental runner
├── api/                 # Module 8 — FastAPI backend
├── dashboard/           # Module 9 — Streamlit UI
└── docs/                 # architecture notes
```
