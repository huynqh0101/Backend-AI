from py2neo import Graph
import pandas as pd

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "huy1552004"
graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), name="yhct")

def verify_kg():
    print("=" * 60)
    print("🔍 KIỂM TRA KNOWLEDGE GRAPH")
    print("=" * 60)
    
    # 1. Đếm nodes và relationships
    node_count = graph.run("MATCH (n) RETURN count(n) AS count").data()[0]['count']
    rel_count = graph.run("MATCH ()-[r]->() RETURN count(r) AS count").data()[0]['count']
    
    print(f"\n📊 Tổng quan:")
    print(f"   - Tổng nodes: {node_count}")
    print(f"   - Tổng relationships: {rel_count}")
    
    # 2. Phân bố node types
    node_types = graph.run("""
        MATCH (n)
        RETURN labels(n)[0] AS type, count(*) AS count
        ORDER BY count DESC
    """).data()
    
    print(f"\n📦 Phân bố node types:")
    for item in node_types:
        print(f"   - {item['type']}: {item['count']}")
    
    # 3. Phân bố relationship types
    rel_types = graph.run("""
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(*) AS count
        ORDER BY count DESC
    """).data()
    
    print(f"\n🔗 Phân bố relationship types:")
    for item in rel_types:
        print(f"   - {item['type']}: {item['count']}")
    
    # 4. Kiểm tra orphan nodes
    orphan_count = graph.run("""
        MATCH (n)
        WHERE NOT (n)-[]-()
        RETURN count(n) AS count
    """).data()[0]['count']
    
    print(f"\n⚠️  Orphan nodes (không có relationship): {orphan_count}")
    
    # 5. Kiểm tra nguyên liệu có link tới cây thuốc
    linked_ingredients = graph.run("""
        MATCH (nl:NGUYÊN_LIỆU)-[:LÀ]->(ct:CÂY_THUỐC)
        RETURN count(*) AS count
    """).data()[0]['count']
    
    total_ingredients = graph.run("""
        MATCH (nl:NGUYÊN_LIỆU)
        RETURN count(*) AS count
    """).data()[0]['count']
    
    print(f"\n🔗 Liên kết NGUYÊN LIỆU <-> CÂY THUỐC:")
    print(f"   - Tổng nguyên liệu: {total_ingredients}")
    print(f"   - Có link tới cây thuốc: {linked_ingredients}")
    if total_ingredients > 0:
        print(f"   - Tỷ lệ: {linked_ingredients/total_ingredients*100:.1f}%")
    else:
        print(f"   - Tỷ lệ: N/A (không có nguyên liệu)")
    
    # 6. Top cây thuốc được dùng nhiều nhất
    top_herbs = graph.run("""
        MATCH (ct:CÂY_THUỐC)<-[:LÀ]-(nl:NGUYÊN_LIỆU)<-[:CHỨA_NGUYÊN_LIỆU]-(bt:BÀI_THUỐC)
        RETURN ct.tên_chính AS cây_thuốc, count(DISTINCT bt) AS số_bài_thuốc
        ORDER BY số_bài_thuốc DESC
        LIMIT 5
    """).data()
    
    print(f"\n🌿 Top 5 cây thuốc được dùng nhiều nhất:")
    for item in top_herbs:
        print(f"   - {item['cây_thuốc']}: {item['số_bài_thuốc']} bài thuốc")
    
    # 7. Kiểm tra bài thuốc thiếu thông tin
    missing_disease = graph.run("""
        MATCH (bt:BÀI_THUỐC)
        WHERE NOT (bt)-[:CHỮA_TRỊ]->()
        RETURN count(*) AS count
    """).data()[0]['count']
    
    missing_ingredients = graph.run("""
        MATCH (bt:BÀI_THUỐC)
        WHERE NOT (bt)-[:CHỨA_NGUYÊN_LIỆU]->()
        RETURN count(*) AS count
    """).data()[0]['count']
    
    print(f"\n⚠️  Bài thuốc thiếu thông tin:")
    print(f"   - Thiếu chỉ định bệnh: {missing_disease}")
    print(f"   - Thiếu nguyên liệu: {missing_ingredients}")
    
    print("\n" + "=" * 60)
    print("✅ HOÀN THÀNH KIỂM TRA")
    print("=" * 60)

if __name__ == "__main__":
    verify_kg()