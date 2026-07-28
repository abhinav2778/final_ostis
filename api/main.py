import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "storage", "chroma_db")
COLLECTION_NAME = "ostis_collection"

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")

resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("OSTIS API starting up...")

    print("Loading embedding model...")
    resources["model"] = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    resources["collection"] = chroma_client.get_or_create_collection(COLLECTION_NAME)

    print("Connecting to Neo4j...")
    resources["neo4j"] = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        resources["neo4j"].verify_connectivity()
        resources["neo4j_ok"] = True
        print("Neo4j connected.")
    except Exception as e:
        resources["neo4j_ok"] = False
        print(f"WARNING: Neo4j not reachable at startup ({e}). Graph endpoints will report errors until it's up.")

    print("OSTIS API ready.\n")
    yield

    print("OSTIS API shutting down...")
    resources["neo4j"].close()


app = FastAPI(
    title="OSTIS API",
    description="Organization-Specific Threat Intelligence System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    industry: str = "All Industries"
    top_k: int = 5


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "loaded",
        "chromadb": "connected",
        "neo4j": "connected" if resources.get("neo4j_ok") else "unavailable",
        "version": "1.0.0",
    }


@app.post("/search")
def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query_embedding = resources["model"].encode([request.query]).tolist()

    where_filter = None
    if request.industry and request.industry != "All Industries":
        where_filter = {"primary_industry": {"$eq": request.industry}}

    results = resources["collection"].query(
        query_embeddings=query_embedding,
        n_results=request.top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = round(max(0, 1 - dist / 2), 3)
        output.append({
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "snippet": doc[:300],
            "similarity_score": similarity,
            "threat_score": meta.get("threat_score", ""),
            "primary_industry": meta.get("primary_industry", ""),
        })

    return {"query": request.query, "results": output, "count": len(output)}


@app.get("/graph")
def get_graph(industry: str = "All", limit: int = 100):
    try:
        with resources["neo4j"].session() as session:
            if industry == "All":
                query = """
                    MATCH (a)-[r]->(b)
                    RETURN a, type(r) AS rel, b
                    LIMIT $limit
                """
                result = session.run(query, limit=limit)
            else:
                query = """
                    MATCH (a)-[r]->(b)
                    WHERE (a:Industry AND a.name = $industry)
                       OR (b:Industry AND b.name = $industry)
                       OR (a:Article AND a.industry = $industry)
                    RETURN a, type(r) AS rel, b
                    LIMIT $limit
                """
                result = session.run(query, industry=industry, limit=limit)

            edges = []
            for record in result:
                node_a, node_b, rel = record["a"], record["b"], record["rel"]
                label_a = list(node_a.labels)[0] if node_a.labels else "Node"
                label_b = list(node_b.labels)[0] if node_b.labels else "Node"
                name_a = (node_a.get("name") or node_a.get("id") or node_a.get("address")
                          or (node_a.get("title", "")[:40]) or "Unknown")
                name_b = (node_b.get("name") or node_b.get("id") or node_b.get("address")
                          or (node_b.get("title", "")[:40]) or "Unknown")
                edges.append({
                    "source": name_a, "source_type": label_a,
                    "target": name_b, "target_type": label_b,
                    "relationship": rel,
                })

        return {"edges": edges, "count": len(edges)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    try:
        total_articles = resources["collection"].count()

        with resources["neo4j"].session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            industry_count = session.run("MATCH (i:Industry) RETURN count(i) AS c").single()["c"]

        return {
            "total_articles": total_articles,
            "industries_covered": industry_count,
            "embedding_dimensions": 384,
            "graph_nodes": node_count,
            "graph_relationships": rel_count,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entities")
def get_entities(industry: str = "All Industries", limit: int = 10):
    try:
        with resources["neo4j"].session() as session:
            if industry == "All Industries":
                malware_q = """
                    MATCH (m:Malware)<-[:MENTIONS]-(a:Article)
                    RETURN m.name AS name, count(a) AS connections
                    ORDER BY connections DESC LIMIT $limit
                """
                cve_q = """
                    MATCH (c:CVE)<-[:MENTIONS]-(a:Article)
                    RETURN c.id AS name, count(a) AS connections
                    ORDER BY connections DESC LIMIT $limit
                """
                actor_q = """
                    MATCH (t:ThreatActor)<-[:MENTIONS]-(a:Article)
                    RETURN t.name AS name, count(a) AS connections
                    ORDER BY connections DESC LIMIT $limit
                """
                params = {"limit": limit}
            else:
                malware_q = """
                    MATCH (m:Malware)<-[:MENTIONS]-(a:Article {industry: $industry})
                    RETURN m.name AS name, count(a) AS connections
                    ORDER BY connections DESC LIMIT $limit
                """
                cve_q = """
                    MATCH (c:CVE)<-[:MENTIONS]-(a:Article {industry: $industry})
                    RETURN c.id AS name, count(a) AS connections
                    ORDER BY connections DESC LIMIT $limit
                """
                actor_q = """
                    MATCH (t:ThreatActor)<-[:MENTIONS]-(a:Article {industry: $industry})
                    RETURN t.name AS name, count(a) AS connections
                    ORDER BY connections DESC LIMIT $limit
                """
                params = {"industry": industry, "limit": limit}

            malware = [dict(r) for r in session.run(malware_q, params)]
            cves = [dict(r) for r in session.run(cve_q, params)]
            actors = [dict(r) for r in session.run(actor_q, params)]

        return {"malware": malware, "cves": cves, "threat_actors": actors}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/industry-stats")
def get_industry_stats():
    try:
        with resources["neo4j"].session() as session:
            result = session.run("""
                MATCH (a:Article)-[:TARGETS]->(i:Industry)
                RETURN i.name AS industry, count(a) AS count
                ORDER BY count DESC
            """)
            stats = [dict(r) for r in result]
        return {"industry_stats": stats}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts")
def get_alerts(limit: int = 10, industry: str = "All Industries"):
    try:
        where_filter = None
        if industry != "All Industries":
            where_filter = {"primary_industry": {"$eq": industry}}

        results = resources["collection"].get(
            where=where_filter,
            include=["metadatas"],
            limit=2000,
        )

        alerts = []
        for meta in results["metadatas"]:
            try:
                score = float(meta.get("threat_score", 0))
            except (TypeError, ValueError):
                score = 0.0
            alerts.append({
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "primary_industry": meta.get("primary_industry", ""),
                "threat_score": score,
            })

        alerts.sort(key=lambda a: a["threat_score"], reverse=True)
        return {"alerts": alerts[:limit], "count": min(limit, len(alerts))}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
