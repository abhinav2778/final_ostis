# ─────────────────────────────────────────────────────────────────
# OSTIS — Utility: quick ChromaDB sanity check
# ─────────────────────────────────────────────────────────────────

import os
import chromadb

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "storage", "chroma_db")
COLLECTION_NAME = "ostis_collection"


def run():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    print(f"Collection '{COLLECTION_NAME}' total count: {collection.count()}")

    results = collection.get(limit=3, include=["metadatas", "embeddings"])
    print("\nSample IDs:", results["ids"])
    print("\nSample metadata:", results["metadatas"])
    if results["embeddings"] is not None and len(results["embeddings"]) > 0:
        print("\nFirst embedding length:", len(results["embeddings"][0]))
        print("First embedding sample:", results["embeddings"][0][:5])


if __name__ == "__main__":
    run()
