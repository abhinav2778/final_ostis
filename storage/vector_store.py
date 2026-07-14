# ─────────────────────────────────────────────────────────────────
# OSTIS — Module 4: Embeddings + Vector Storage
# Input  : classifier/tagged_articles.csv
# Output : ChromaDB persistent collection with industry metadata
#
# Model: all-MiniLM-L6-v2 (384-dim) — small, fast, offline-capable,
# purpose-built for semantic similarity rather than generation.
# ─────────────────────────────────────────────────────────────────

import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV  = os.path.join(BASE_DIR, "classifier", "tagged_articles.csv")
CHROMA_DIR = os.path.join(BASE_DIR, "storage", "chroma_db")
COLLECTION_NAME = "ostis_collection"
BATCH_SIZE = 50


def run(rebuild: bool = True):
    print("=" * 60)
    print("OSTIS — Module 4: Embeddings + Vector Storage")
    print("=" * 60)

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found. Run classification first (Module 3).")
        return

    print("Loading sentence-transformer model: all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.\n")

    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["clean_text", "title", "url"])
    print(f"Total articles loaded: {len(df)}\n")

    print(f"Connecting to ChromaDB at {CHROMA_DIR} ...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Old collection deleted. Rebuilding fresh.\n")
        except Exception:
            print("No existing collection. Creating fresh.\n")
        collection = client.create_collection(COLLECTION_NAME)
    else:
        collection = client.get_or_create_collection(COLLECTION_NAME)

    total = len(df)
    for i in range(0, total, BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE]
        texts = batch["clean_text"].tolist()
        ids = [f"doc_{i + j}" for j in range(len(batch))]
        embeddings = model.encode(texts).tolist()

        metadatas = [{
            "title":              str(row.get("title", "")),
            "url":                str(row.get("url", "")),
            "threat_score":       str(row.get("threat_score", "")),
            "primary_industry":   str(row.get("primary_industry", "")),
            "industry_score":     str(row.get("industry_score", "")),
        } for _, row in batch.iterrows()]

        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        print(f"  Stored {min(i + BATCH_SIZE, total)}/{total} articles...")

    print(f"\n{'=' * 60}")
    print("VECTOR STORE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total articles embedded : {total}")
    print(f"ChromaDB location       : {CHROMA_DIR}")
    print(f"Collection name         : {COLLECTION_NAME}")
    print(f"Embedding dimensions    : 384")
    print(f"Collection count now    : {collection.count()}")


if __name__ == "__main__":
    run()
