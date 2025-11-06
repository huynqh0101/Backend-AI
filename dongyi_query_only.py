# Hệ thống Truy vấn Kiến thức Đông y - CHỈ NEO4J
# ------------------------------------------------
import os
import asyncio
import logging
import traceback
from neo4j import GraphDatabase

# --- Cấu hình ---
print("--- Hệ thống Truy vấn Kiến thức Đông y (Neo4j Only) ---")

# Neo4j Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "huy1552004"
NEO4J_DATABASE = "dongyi"

print(f"Đã cấu hình Neo4j (Database: {NEO4J_DATABASE})")

# --- Neo4j Query Helper ---
class DongyiQueryHelper:
    def __init__(self, uri, username, password, database="dongyi"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        
    def close(self):
        self.driver.close()
    
    def query_dongyi_kg(self, query_text):
        """Truy vấn Knowledge Graph Đông y trong Neo4j"""
        try:
            with self.driver.session(database=self.database) as session:
                # Debug
                count_result = session.run("MATCH (n) RETURN count(n) as total")
                total_entities = count_result.single()["total"]
                print(f"   Database có {total_entities} nodes")
                
                if total_entities == 0:
                    print(f"   Database '{self.database}' trống!")
                    return []
                
                print(f"   Tìm kiếm: '{query_text}'")
                
                # Query 1: Tìm bài thuốc chữa bệnh
                query = """
                MATCH (r:`BÀI THUỐC`)-[:`CHỮA TRỊ`]->(b:`BỆNH`)
                WHERE toLower(b.tên_bệnh) CONTAINS toLower($query_text)
                   OR toLower(r.tên_bài_thuốc) CONTAINS toLower($query_text)
                OPTIONAL MATCH (r)-[:`CHỨA NGUYÊN LIỆU`]->(n:`NGUYÊN LIỆU`)
                OPTIONAL MATCH (r)-[:`CÓ CÔNG HIỆU`]->(e:`CÔNG HIỆU`)
                RETURN DISTINCT
                    r.tên_bài_thuốc AS ten_bai_thuoc,
                    b.tên_bệnh AS ten_benh,
                    r.liều_lượng_cách_dùng AS lieu_luong,
                    r.chú_ý AS chu_y,
                    collect(DISTINCT n.tên_nguyên_liệu)[..5] AS nguyen_lieu,
                    collect(DISTINCT e.tên_công_hiệu)[..3] AS cong_hieu
                LIMIT 5
                """
                
                results = session.run(query, query_text=query_text)
                entities = []
                for record in results:
                    ten_bai = record.get('ten_bai_thuoc', 'N/A')
                    ten_benh = record.get('ten_benh', 'N/A')
                    lieu_luong = record.get('lieu_luong', '')
                    chu_y = record.get('chu_y', '')
                    nguyen_lieu = [nl for nl in record.get('nguyen_lieu', []) if nl]
                    cong_hieu = [ch for ch in record.get('cong_hieu', []) if ch]
                    
                    description = f"**Chữa bệnh:** {ten_benh}\n"
                    if nguyen_lieu:
                        description += f"**Nguyên liệu:** {', '.join(nguyen_lieu)}\n"
                    if cong_hieu:
                        description += f"**Công hiệu:** {', '.join(cong_hieu)}\n"
                    if lieu_luong:
                        description += f"**Liều lượng:** {lieu_luong[:300]}...\n"
                    if chu_y:
                        description += f"**Chú ý:** {chu_y[:200]}..."
                    
                    entities.append({
                        'ten_bai_thuoc': ten_bai,
                        'description': description
                    })
                
                if entities:
                    print(f"   ✓ Tìm thấy {len(entities)} bài thuốc")
                    return entities
                
                # Query 2: Tìm theo nguyên liệu
                query2 = """
                MATCH (r:`BÀI THUỐC`)-[:`CHỨA NGUYÊN LIỆU`]->(n:`NGUYÊN LIỆU`)
                WHERE toLower(n.tên_nguyên_liệu) CONTAINS toLower($query_text)
                OPTIONAL MATCH (r)-[:`CHỮA TRỊ`]->(b:`BỆNH`)
                RETURN DISTINCT
                    r.tên_bài_thuốc AS ten_bai_thuoc,
                    n.tên_nguyên_liệu AS nguyen_lieu,
                    collect(DISTINCT b.tên_bệnh)[..3] AS benh
                LIMIT 5
                """
                
                results = session.run(query2, query_text=query_text)
                entities = []
                for record in results:
                    ten_bai = record.get('ten_bai_thuoc', 'N/A')
                    nguyen_lieu = record.get('nguyen_lieu', 'N/A')
                    benh = [b for b in record.get('benh', []) if b]
                    
                    description = f"**Nguyên liệu:** {nguyen_lieu}\n"
                    if benh:
                        description += f"**Chữa bệnh:** {', '.join(benh)}"
                    
                    entities.append({
                        'ten_bai_thuoc': ten_bai,
                        'description': description
                    })
                
                if entities:
                    print(f"   ✓ Tìm thấy {len(entities)} bài thuốc")
                    return entities
                
                # Query 3: Tìm theo công hiệu
                query3 = """
                MATCH (r:`BÀI THUỐC`)-[:`CÓ CÔNG HIỆU`]->(e:`CÔNG HIỆU`)
                WHERE toLower(e.tên_công_hiệu) CONTAINS toLower($query_text)
                RETURN DISTINCT
                    r.tên_bài_thuốc AS ten_bai_thuoc,
                    e.tên_công_hiệu AS cong_hieu
                LIMIT 5
                """
                
                results = session.run(query3, query_text=query_text)
                entities = []
                for record in results:
                    entities.append({
                        'ten_bai_thuoc': record.get('ten_bai_thuoc', 'N/A'),
                        'description': f"**Công hiệu:** {record.get('cong_hieu', 'N/A')}"
                    })
                
                if entities:
                    print(f"   ✓ Tìm thấy {len(entities)} bài thuốc")
                    return entities
                
                print("   ✗ Không tìm thấy kết quả")
                return []
                
        except Exception as e:
            print(f"Lỗi truy vấn Neo4j: {e}")
            traceback.print_exc()
            return []

async def interactive_dongyi_query():
    """Chế độ truy vấn tương tác - CHỈ NEO4J"""
    print("\n=== CHƯƠNG TRÌNH TRUY VẤN ĐÔNG Y (NEO4J) ===")
    print("Nhập 'exit' để thoát, 'help' để xem hướng dẫn\n")
    
    # Khởi tạo Neo4j
    try:
        neo4j_helper = DongyiQueryHelper(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)
        
        # Test connection
        with neo4j_helper.driver.session(database=neo4j_helper.database) as session:
            session.run("RETURN 1")
        print("✓ Kết nối Neo4j thành công\n")
        
    except Exception as e:
        print(f"✗ Không kết nối được Neo4j: {e}")
        return
    
    try:
        while True:
            try:
                print("="*60)
                user_query = input("Nhập câu hỏi về Đông y: ").strip()
                
                if user_query.lower() == 'exit':
                    print("\nTạm biệt!")
                    break
                elif user_query.lower() == 'help':
                    print_help()
                    continue
                elif not user_query:
                    print("Vui lòng nhập câu hỏi!")
                    continue
                
                print(f"\nĐang tìm kiếm: '{user_query}'...\n")
                
                # Truy vấn Neo4j
                entities = neo4j_helper.query_dongyi_kg(user_query)
                
                if entities:
                    print(f"\n📋 KẾT QUẢ TÌM KIẾM:\n")
                    for i, entity in enumerate(entities, 1):
                        print(f"{'─'*60}")
                        print(f"🔹 BÀI THUỐC {i}: {entity['ten_bai_thuoc']}")
                        print(f"{'─'*60}")
                        print(entity['description'])
                        print()
                else:
                    print("\n✗ Không tìm thấy kết quả")
                    print("💡 Thử từ khóa: 'sốt', 'ho', 'đau đầu', 'lá tre', 'thạch cao'\n")
                
            except KeyboardInterrupt:
                print("\n\nTạm biệt!")
                break
            except Exception as e:
                print(f"Lỗi: {e}")
                traceback.print_exc()
        
    finally:
        neo4j_helper.close()
        print("✓ Đã đóng kết nối Neo4j")

def print_help():
    """In hướng dẫn"""
    print("\n" + "="*60)
    print("📖 HƯỚNG DẪN SỬ DỤNG")
    print("="*60)
    print("• Nhập câu hỏi về Đông y để tìm kiếm bài thuốc")
    print("• 'help' - Xem hướng dẫn")
    print("• 'exit' - Thoát chương trình")
    print("\n💡 VÍ DỤ CÂU HỎI:")
    print("   - Bài thuốc chữa sốt")
    print("   - Thuốc nào có lá tre")
    print("   - Công hiệu thanh nhiệt")
    print("   - Chữa ho")
    print("="*60 + "\n")

async def main():
    """Hàm chính"""
    print("\n" + "="*60)
    print("🏥 HỆ THỐNG TRA CỨU KIẾN THỨC ĐÔNG Y")
    print("="*60)
    print(f"📊 Database: {NEO4J_DATABASE}")
    print("="*60 + "\n")
    
    try:
        await interactive_dongyi_query()
    except Exception as e:
        print(f"Lỗi: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.ERROR)
        asyncio.run(main())
        print("\n✓ Chương trình hoàn tất!\n")
    except KeyboardInterrupt:
        print("\n⚠️  Dừng bởi người dùng\n")
    except Exception as e:
        print(f"\n✗ Lỗi: {e}\n")