import os
from dongyi_lightrag import DongyiKnowledgeGraph, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE, ensure_vietnamese

WORKING_DIR = "./lightrag_dongyi_neo4j"
graphml_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")

if not os.path.exists(graphml_file):
    print(f"⚠️  Không tìm thấy GraphML: {graphml_file}")
    exit(1)

dongyi_kg = DongyiKnowledgeGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)

choice = input("\n⚠️  Xóa dữ liệu cũ trong Neo4j? (y/n): ").strip().lower()
if choice == 'y':
    dongyi_kg.clear_database()

dongyi_kg.import_from_graphml(graphml_file)
dongyi_kg.get_stats()

print(f"\n📊 Xem trong Neo4j Browser: http://localhost:7474")
print(f"   :use {NEO4J_DATABASE}")
print(f"   MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 25") 