# Module 4 — Embeddings + Vector Storage

Generates semantic embeddings for classified articles and stores them in
ChromaDB for similarity search.

## Run

```bash
python storage/vector_store.py
```

Input: `classifier/tagged_articles.csv`
Output: persistent ChromaDB at `storage/chroma_db/`, collection name
`ostis_collection`.

## Design notes

- **Model**: `all-MiniLM-L6-v2`, 384 dimensions. Chosen for size (~80MB)
  and speed over larger embedding models — this is a retrieval task, not
  generation.
- **Rebuild vs. incremental**: `run(rebuild=True)` (the default) deletes
  and recreates the collection from scratch — use this after re-running
  classification on the full corpus. The incremental pipeline (Module 7)
  uses `rebuild=False` and only adds new documents by ID, checking
  existing IDs first to avoid duplicate embeddings.
- **Metadata stored per document**: title, url, threat_score,
  primary_industry, industry_score — this is what the dashboard's
  industry filter and search result cards read directly.
- **Batching**: 50 articles per batch to keep memory usage predictable
  on modest hardware.
