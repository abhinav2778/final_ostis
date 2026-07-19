# Module 6 — Knowledge Graph (Neo4j)

Builds a graph of relationships between articles, threat actors, malware,
CVEs, IP addresses, and target industries.

## Schema

Article -[:TARGETS]-> Industry
Article -[:MENTIONS]-> CVE / Malware / ThreatActor / IPAddress
ThreatActor -[:USES]-> Malware
Malware -[:EXPLOITS]-> CVE
ThreatActor -[:TARGETS]-> Industry
ThreatActor -[:USES_INFRASTRUCTURE]-> IPAddress

## Setup

1. Open Neo4j Desktop, create/start a local database.
2. Set the password in .env (NEO4J_PASSWORD=...) to match Neo4j Desktop.
3. Default bolt URI is neo4j://127.0.0.1:7687, user neo4j.

## Run

python graph/build_graph.py     # builds the graph (clears + rebuilds)
python graph/check_graph.py     # prints node/relationship counts

Input: ner/ner_output.csv
View the graph directly at http://localhost:7474 (Neo4j Browser).
