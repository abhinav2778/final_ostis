# ─────────────────────────────────────────────────────────────────
# OSTIS — Module 7: Incremental Pipeline Runner
#
# Detects articles in storage/articles.jsonl that haven't been processed
# yet (tracked via storage/processed_urls.txt), and runs only those
# through preprocessing -> classification -> NER -> embedding -> graph
# update. Existing data (ChromaDB, Neo4j) is never wiped -- everything
# uses MERGE/upsert semantics so re-running is always safe.
#
# Deliberately reuses the already-fixed core functions instead of
# reimplementing simplified versions inline:
#   - preprocessing.preprocess.clean_article_text  (CVE/IP-safe cleaning)
#   - ner.extract_entities.extract_entities         (word-boundary +
#     ambiguous-term-aware NER -- reimplementing this inline would risk
#     silently reintroducing the "conti"/"play" false-positive bugs)
#
# Requires: fine-tuned classifier models must already exist
# (classifier/finetuned/binary, classifier/finetuned/industry).
#
# Run: python pipeline/incremental_pipeline.py
# ─────────────────────────────────────────────────────────────────

import os
import sys
import json
import hashlib
from datetime import datetime

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for subdir in ("preprocessing", "classifier", "ner"):
    sys.path.insert(0, os.path.join(BASE_DIR, subdir))

from preprocess import clean_article_text
from extract_entities import extract_entities
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

ARTICLES_JSONL = os.path.join(BASE_DIR, "storage", "articles.jsonl")
LEDGER_FILE = os.path.join(BASE_DIR, "storage", "processed_urls.txt")
CHROMA_PATH = os.path.join(BASE_DIR, "storage", "chroma_db")
BINARY_DIR = os.path.join(BASE_DIR, "classifier", "finetuned", "binary")
INDUSTRY_DIR = os.path.join(BASE_DIR, "classifier", "finetuned", "industry")
COLLECTION_NAME = "ostis_collection"

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")

MIN_WORDS = 100
THREAT_THRESHOLD_LABEL = 1


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_ledger():
    if not os.path.exists(LEDGER_FILE):
        return set()
    with open(LEDGER_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def append_ledger(urls):
    os.makedirs(os.path.dirname(LEDGER_FILE), exist_ok=True)
    with open(LEDGER_FILE, "a", encoding="utf-8") as f:
        for url in urls:
            f.write(url + "\n")


def load_new_articles(seen_urls):
    if not os.path.exists(ARTICLES_JSONL):
        return []
    new_articles = []
    with open(ARTICLES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                article = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = article.get("url", "")
            if url and url not in seen_urls:
                new_articles.append(article)
    return new_articles


def predict(text, tokenizer, model, device, max_length=256):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_class = int(probs.argmax())
    confidence = float(probs.max())
    return pred_class, confidence


def upsert_article_to_graph(driver, article: dict):
    def run_query(query, params=None):
        with driver.session() as session:
            session.run(query, params or {})

    url = article["url"]
    title = article["title"]
    industry = article["primary_industry"]
    threat_score = str(article["threat_score"])
    cves, ips, malware, actors = article["cves"], article["ips"], article["malware"], article["actors"]

    if not any([cves, ips, malware, actors]):
        return False

    run_query("""
        MERGE (a:Article {url: $url})
        SET a.title = $title, a.industry = $industry, a.threat_score = $threat_score
    """, {"url": url, "title": title, "industry": industry, "threat_score": threat_score})

    if industry:
        run_query("""
            MERGE (i:Industry {name: $industry})
            WITH i MATCH (a:Article {url: $url}) MERGE (a)-[:TARGETS]->(i)
        """, {"industry": industry, "url": url})

    for cve in cves:
        run_query("""
            MERGE (c:CVE {id: $cve})
            WITH c MATCH (a:Article {url: $url}) MERGE (a)-[:MENTIONS]->(c)
        """, {"cve": cve, "url": url})

    for m in malware:
        run_query("""
            MERGE (m:Malware {name: $m})
            WITH m MATCH (a:Article {url: $url}) MERGE (a)-[:MENTIONS]->(m)
        """, {"m": m, "url": url})

    for actor in actors:
        run_query("""
            MERGE (t:ThreatActor {name: $actor})
            WITH t MATCH (a:Article {url: $url}) MERGE (a)-[:MENTIONS]->(t)
        """, {"actor": actor, "url": url})

    for ip in ips:
        run_query("""
            MERGE (ip:IPAddress {address: $ip})
            WITH ip MATCH (a:Article {url: $url}) MERGE (a)-[:MENTIONS]->(ip)
        """, {"ip": ip, "url": url})

    for actor in actors:
        for m in malware:
            run_query("""
                MERGE (t:ThreatActor {name: $actor}) MERGE (m:Malware {name: $m})
                MERGE (t)-[:USES]->(m)
            """, {"actor": actor, "m": m})

    for m in malware:
        for cve in cves:
            run_query("""
                MERGE (m:Malware {name: $m}) MERGE (c:CVE {id: $cve})
                MERGE (m)-[:EXPLOITS]->(c)
            """, {"m": m, "cve": cve})

    for actor in actors:
        if industry:
            run_query("""
                MERGE (t:ThreatActor {name: $actor}) MERGE (i:Industry {name: $industry})
                MERGE (t)-[:TARGETS]->(i)
            """, {"actor": actor, "industry": industry})

    for actor in actors:
        for ip in ips:
            run_query("""
                MERGE (t:ThreatActor {name: $actor}) MERGE (ip:IPAddress {address: $ip})
                MERGE (t)-[:USES_INFRASTRUCTURE]->(ip)
            """, {"actor": actor, "ip": ip})

    return True


def run():
    print("=" * 60)
    print("OSTIS — Module 7: Incremental Pipeline Runner")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    print("\nStep 1: Checking ledger for new articles...")
    seen_urls = load_ledger()
    print(f"Already processed: {len(seen_urls)} articles")

    new_articles = load_new_articles(seen_urls)
    print(f"New articles found: {len(new_articles)}")

    if not new_articles:
        print("\nNo new articles. Pipeline up to date.")
        return

    print("\nStep 2: Preprocessing new articles...")
    clean_articles = []
    all_new_urls = [a.get("url", "") for a in new_articles]

    for art in new_articles:
        clean_text = clean_article_text(art.get("content", ""))
        if len(clean_text.split()) >= MIN_WORDS and art.get("title", "").strip():
            clean_articles.append({
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "clean_text": clean_text,
            })
    print(f"After quality filter: {len(clean_articles)} articles")

    if not clean_articles:
        append_ledger(all_new_urls)
        print("No articles passed quality filter. Ledger updated, nothing else to do.")
        return

    print("\nStep 3: Classifying (fine-tuned CySecBERT)...")
    if not os.path.exists(BINARY_DIR) or not os.path.exists(INDUSTRY_DIR):
        print(f"ERROR: fine-tuned models not found. Run classifier/finetune_binary.py")
        print("and classifier/finetune_industry.py first.")
        return

    binary_tok = AutoTokenizer.from_pretrained(BINARY_DIR)
    binary_model = AutoModelForSequenceClassification.from_pretrained(BINARY_DIR).to(device)
    binary_model.eval()

    industry_tok = AutoTokenizer.from_pretrained(INDUSTRY_DIR)
    industry_model = AutoModelForSequenceClassification.from_pretrained(INDUSTRY_DIR).to(device)
    industry_model.eval()

    with open(os.path.join(INDUSTRY_DIR, "label_map.json")) as f:
        label_map = json.load(f)

    tagged = []
    for art in clean_articles:
        input_text = f"{art['title']}. {art['clean_text'][:300]}"
        binary_pred, binary_conf = predict(input_text, binary_tok, binary_model, device)
        if binary_pred != THREAT_THRESHOLD_LABEL:
            continue
        industry_pred, industry_conf = predict(input_text, industry_tok, industry_model, device)
        tagged.append({
            "title": art["title"],
            "url": art["url"],
            "clean_text": art["clean_text"],
            "threat_score": round(binary_conf, 4),
            "primary_industry": label_map[str(industry_pred)],
            "industry_score": round(industry_conf, 4),
        })
    print(f"Threat-relevant new articles: {len(tagged)}")

    if not tagged:
        append_ledger(all_new_urls)
        print("No threats found in new articles. Ledger updated.")
        return

    print("\nStep 4: Extracting entities...")
    for art in tagged:
        cves, ips, malware, actors = extract_entities(art["clean_text"])
        art["cves"], art["ips"], art["malware"], art["actors"] = cves, ips, malware, actors
    print(f"Entity extraction complete for {len(tagged)} articles.")

    print("\nStep 5: Embedding into ChromaDB...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    existing_ids = set(collection.get(include=[])["ids"])
    ids = [hashlib.md5(a["url"].encode()).hexdigest() for a in tagged]
    to_embed = [(a, i) for a, i in zip(tagged, ids) if i not in existing_ids]

    if to_embed:
        texts = [a["clean_text"] for a, _ in to_embed]
        embed_ids = [i for _, i in to_embed]
        metadatas = [{
            "title": a["title"], "url": a["url"],
            "threat_score": str(a["threat_score"]),
            "primary_industry": a["primary_industry"],
            "industry_score": str(a["industry_score"]),
        } for a, _ in to_embed]
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
        collection.add(ids=embed_ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        print(f"Embedded {len(to_embed)} new articles (skipped {len(tagged) - len(to_embed)} already present).")
    else:
        print("All articles already embedded.")
    print(f"ChromaDB total: {collection.count()} articles")

    print("\nStep 6: Updating knowledge graph...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        updated = sum(1 for art in tagged if upsert_article_to_graph(driver, art))
        driver.close()
        print(f"Knowledge graph updated with {updated} articles (of {len(tagged)} tagged).")
    except Exception as e:
        print(f"WARNING: knowledge graph update failed: {e}")
        print("ChromaDB was updated successfully; graph can be rebuilt manually later.")

    print("\nStep 7: Updating ledger...")
    append_ledger(all_new_urls)
    print(f"Ledger updated. Total processed: {len(seen_urls) + len(all_new_urls)}")

    print("\n" + "=" * 60)
    print("OSTIS Incremental Pipeline — Complete")
    print(f"  New articles found   : {len(new_articles)}")
    print(f"  After quality filter : {len(clean_articles)}")
    print(f"  Threat-relevant      : {len(tagged)}")
    print(f"  ChromaDB total now   : {collection.count()}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    run()
