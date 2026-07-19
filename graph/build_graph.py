import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "ner", "ner_output.csv")

NEO4J_URI      = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")


def parse_list(value: str):
    if not isinstance(value, str) or not value.strip():
        return []
    return [v.strip() for v in value.split(",") if v.strip() and v.strip().lower() != "nan"]


def run(clear_first: bool = True):
    print("=" * 60)
    print("OSTIS — Module 6: Knowledge Graph Builder")
    print("=" * 60)

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found. Run NER first (Module 5).")
        return

    print(f"Connecting to Neo4j at {NEO4J_URI} ...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Connected.\n")
    except Exception as e:
        print(f"ERROR: could not connect to Neo4j: {e}")
        print("Make sure Neo4j Desktop is running and .env has the correct password.")
        return

    def run_query(query, params=None):
        with driver.session() as session:
            session.run(query, params or {})

    if clear_first:
        print("Clearing existing graph...")
        run_query("MATCH (n) DETACH DELETE n")

    df = pd.read_csv(INPUT_CSV).fillna("")
    print(f"Articles loaded: {len(df)}\n")

    articles_added = 0
    nodes_created = 0
    relations_added = 0

    print("Building knowledge graph...")
    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        url = str(row.get("url", "")).strip()
        industry = str(row.get("primary_industry", "")).strip()
        threat_score = str(row.get("threat_score", "")).strip()

        cves = parse_list(row.get("cve_ids", ""))
        malwares = parse_list(row.get("malware_names", ""))
        actors = parse_list(row.get("threat_actors", ""))
        ips = parse_list(row.get("ip_addresses", ""))

        if not url or not any([cves, malwares, actors, ips]):
            continue

        run_query("""
            MERGE (a:Article {url: $url})
            SET a.title = $title, a.industry = $industry, a.threat_score = $threat_score
        """, {"url": url, "title": title, "industry": industry, "threat_score": threat_score})
        articles_added += 1

        if industry:
            run_query("""
                MERGE (i:Industry {name: $industry})
                WITH i
                MATCH (a:Article {url: $url})
                MERGE (a)-[:TARGETS]->(i)
            """, {"industry": industry, "url": url})

        for cve in cves:
            run_query("""
                MERGE (c:CVE {id: $cve})
                WITH c
                MATCH (a:Article {url: $url})
                MERGE (a)-[:MENTIONS]->(c)
            """, {"cve": cve, "url": url})
            nodes_created += 1

        for malware in malwares:
            run_query("""
                MERGE (m:Malware {name: $malware})
                WITH m
                MATCH (a:Article {url: $url})
                MERGE (a)-[:MENTIONS]->(m)
            """, {"malware": malware, "url": url})
            nodes_created += 1

        for actor in actors:
            run_query("""
                MERGE (t:ThreatActor {name: $actor})
                WITH t
                MATCH (a:Article {url: $url})
                MERGE (a)-[:MENTIONS]->(t)
            """, {"actor": actor, "url": url})
            nodes_created += 1

        for ip in ips:
            run_query("""
                MERGE (ip:IPAddress {address: $ip})
                WITH ip
                MATCH (a:Article {url: $url})
                MERGE (a)-[:MENTIONS]->(ip)
            """, {"ip": ip, "url": url})
            nodes_created += 1

        for actor in actors:
            for malware in malwares:
                run_query("""
                    MERGE (t:ThreatActor {name: $actor})
                    MERGE (m:Malware {name: $malware})
                    MERGE (t)-[:USES]->(m)
                """, {"actor": actor, "malware": malware})
                relations_added += 1

        for malware in malwares:
            for cve in cves:
                run_query("""
                    MERGE (m:Malware {name: $malware})
                    MERGE (c:CVE {id: $cve})
                    MERGE (m)-[:EXPLOITS]->(c)
                """, {"malware": malware, "cve": cve})
                relations_added += 1

        for actor in actors:
            if industry:
                run_query("""
                    MERGE (t:ThreatActor {name: $actor})
                    MERGE (i:Industry {name: $industry})
                    MERGE (t)-[:TARGETS]->(i)
                """, {"actor": actor, "industry": industry})
                relations_added += 1

        for actor in actors:
            for ip in ips:
                run_query("""
                    MERGE (t:ThreatActor {name: $actor})
                    MERGE (ip:IPAddress {address: $ip})
                    MERGE (t)-[:USES_INFRASTRUCTURE]->(ip)
                """, {"actor": actor, "ip": ip})
                relations_added += 1

    driver.close()

    print(f"\n{'─' * 40}")
    print(f"Articles added      : {articles_added}")
    print(f"Entity nodes added  : {nodes_created}")
    print(f"Relationships added : {relations_added}")
    print(f"{'─' * 40}")
    print("View at: http://localhost:7474")
    print("Knowledge graph built successfully.")


if __name__ == "__main__":
    run()
