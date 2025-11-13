# Hệ thống LightRAG + Neo4j cho Kiến thức Đông y
# ------------------------------------------------
# Dựng và nạp các bài thuốc Đông y vào LightRAG và Neo4j

import os
import asyncio
import logging
import traceback
import xml.etree.ElementTree as ET
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from lightrag.kg.shared_storage import initialize_pipeline_status
from neo4j import GraphDatabase
import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# --- Cấu hình ---
print("--- Hệ thống LightRAG + Neo4j cho Kiến thức Đông y ---")

# Gemini API
GEMINI_API_KEY = "AIzaSyAfExWuv7945whyX7klFFEnjGDcFeDSxBA"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Neo4j Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "huy1552004"
NEO4J_DATABASE = "dongyi"  # Database chuyên về Đông y

# Working Directory
WORKING_DIR = "./dongyi_knowledge_graph"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)
    print(f"Đã tạo thư mục lưu trữ kiến thức: {WORKING_DIR}")

print(f"Đã cấu hình Gemini API và Neo4j (Database: {NEO4J_DATABASE})")

# --- Neo4j Knowledge Graph cho Đông y ---
class DongyiKnowledgeGraph:
    def __init__(self, uri, username, password, database="dongyi"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        print(f"Kết nối Neo4j database Đông y: {database}")
        
    def close(self):
        self.driver.close()
    
    def create_entity(self, tx, entity_id, entity_type, description, source_id):
        """Tạo entity trong Neo4j - SỬ DỤNG MERGE để tránh trùng lặp"""
        query = """
        MERGE (e:Entity {id: $entity_id})
        ON CREATE SET 
            e.type = $entity_type, 
            e.description = $description, 
            e.source_id = $source_id,
            e.displayName = $entity_id,
            e.created_at = datetime()
        ON MATCH SET
            e.type = $entity_type,
            e.description = $description,
            e.updated_at = datetime()
        RETURN e
        """
        return tx.run(query, entity_id=entity_id, entity_type=entity_type, 
                     description=description, source_id=source_id)
    
    def create_relationship(self, tx, source_id, target_id, weight, description, keywords, source_doc):
        """Tạo relationship trong Neo4j - SỬ DỤNG MERGE để tránh trùng lặp"""
        query = """
        MATCH (source:Entity {id: $source_id})
        MATCH (target:Entity {id: $target_id})
        MERGE (source)-[r:CHUA_TRI]->(target)
        ON CREATE SET
            r.weight = $weight, 
            r.description = $description, 
            r.keywords = $keywords,
            r.source_doc = $source_doc,
            r.created_at = datetime()
        ON MATCH SET
            r.weight = $weight,
            r.description = $description,
            r.keywords = $keywords,
            r.updated_at = datetime()
        RETURN r
        """
        return tx.run(query, source_id=source_id, target_id=target_id, 
                     weight=weight, description=description, 
                     keywords=keywords, source_doc=source_doc)
    
    def import_from_graphml(self, graphml_file):
        """Import GraphML file vào Neo4j - CHẾ ĐỘ BỔ SUNG"""
        try:
            print(f"Đang đọc GraphML file: {graphml_file}")
            tree = ET.parse(graphml_file)
            root = tree.getroot()
            
            # Định nghĩa namespace
            ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
            
            with self.driver.session(database=self.database) as session:
                # Import nodes (entities)
                nodes = root.findall('.//graphml:node', ns)
                print(f"Tìm thấy {len(nodes)} thành phần y học trong GraphML")
                
                entity_count = 0
                entity_updated = 0
                for node in nodes:
                    node_id = node.get('id')
                    entity_type = "Unknown"
                    description = ""
                    
                    # Đọc data của node
                    for data in node.findall('graphml:data', ns):
                        key = data.get('key')
                        if key == 'd1' or key == 'entity_type':
                            entity_type = data.text or "Unknown"
                        elif key == 'd2' or key == 'description':
                            description = data.text or ""
                    
                    result = session.execute_write(self.create_entity, node_id, entity_type, description, "dongyi_import")
                    entity_count += 1
                
                print(f"✅ Đã xử lý {entity_count} thành phần y học (MERGE - tự động tránh trùng)")
                
                # Import edges (relationships)
                edges = root.findall('.//graphml:edge', ns)
                print(f"Tìm thấy {len(edges)} mối liên hệ chữa trị trong GraphML")
                
                rel_count = 0
                for edge in edges:
                    source_id = edge.get('source')
                    target_id = edge.get('target')
                    weight = 1.0
                    description = ""
                    keywords = ""
                    
                    # Đọc data của edge
                    for data in edge.findall('graphml:data', ns):
                        key = data.get('key')
                        if key == 'd5' or key == 'weight':
                            try:
                                weight = float(data.text or 1.0)
                            except:
                                weight = 1.0
                        elif key == 'd6' or key == 'description':
                            description = data.text or ""
                        elif key == 'd7' or key == 'keywords':
                            keywords = data.text or ""
                    
                    session.execute_write(self.create_relationship, source_id, target_id, 
                                        weight, description, keywords, "dongyi_import")
                    rel_count += 1
                
                print(f"✅ Đã xử lý {rel_count} mối liên hệ chữa trị (MERGE - tự động tránh trùng)")
                
        except Exception as e:
            print(f"Lỗi import GraphML: {e}")
            traceback.print_exc()
            raise e
    
    def get_stats(self):
        """Lấy thống kê database"""
        with self.driver.session(database=self.database) as session:
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r:CHUA_TRI]->() RETURN count(r) as count").single()["count"]
            
            print(f"📊 Thống kê Kiến thức Đông y ({self.database}):")
            print(f"   - Thành phần y học: {entity_count}")
            print(f"   - Mối liên hệ chữa trị: {rel_count}")
            return {"entities": entity_count, "relationships": rel_count}

# --- LightRAG Functions ---
async def gemini_llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
    """Custom LLM function sử dụng Gemini API cho Đông y"""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        if history_messages is None:
            history_messages = []

        dongyi_system_prompt = """Bạn là chuyên gia về y học cổ truyền Đông y. 
Hãy trả lời chính xác về các bài thuốc, dược liệu, bệnh lý và phương pháp chữa trị theo Đông y.
Luôn trả lời bằng tiếng Việt."""
        
        combined_prompt = ""
        if system_prompt:
            combined_prompt += f"{dongyi_system_prompt}\n{system_prompt}\n"
        else:
            combined_prompt += f"{dongyi_system_prompt}\n"

        for msg in history_messages:
            combined_prompt += f"{msg['role']}: {msg['content']}\n"

        combined_prompt += f"user: {prompt}"

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[combined_prompt],
            config=types.GenerateContentConfig(max_output_tokens=1000, temperature=0.1),
        )

        return response.text
    except Exception as e:
        print(f"Lỗi Gemini LLM: {e}")
        raise e

async def sentence_transformer_embedding_func(texts):
    """Custom embedding function sử dụng SentenceTransformer"""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        print(f"Lỗi Embedding: {e}")
        raise e

async def initialize_rag():
    """Khởi tạo LightRAG instance cho Đông y"""
    print("\n--- Khởi tạo LightRAG cho Kiến thức Đông y ---")
    
    try:
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=gemini_llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=sentence_transformer_embedding_func,
            ),
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
        )
        
        print("Đang khởi tạo storages...")
        await rag.initialize_storages()
        
        print("Đang khởi tạo pipeline status...")
        await initialize_pipeline_status()
        
        print("✅ LightRAG đã sẵn sàng (chế độ BỔ SUNG)")
        return rag
        
    except Exception as e:
        print(f"Lỗi khởi tạo LightRAG: {e}")
        traceback.print_exc()
        return None

async def load_documents_from_file(file_path):
    """Đọc tài liệu từ file văn bản (phân cách bằng ---)"""
    print(f"\n--- Đọc tài liệu từ file: {file_path} ---")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Tách tài liệu theo dấu ---
        documents = content.split('---')
        documents = [doc.strip() for doc in documents if doc.strip()]
        
        print(f"✅ Đã tải {len(documents)} tài liệu từ file")
        return documents
        
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return []

async def load_documents_from_json(file_path):
    """Đọc tài liệu từ file JSON"""
    print(f"\n--- Đọc tài liệu từ file JSON: {file_path} ---")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format mỗi document thành text
        documents = []
        for item in data:
            doc = f"""
{item.get('name', 'Không có tên')}

Chữa trị: {item.get('chua_tri', 'N/A')}
Thành phần: {item.get('thanh_phan', 'N/A')}
Cách chế biến: {item.get('che_bien', 'N/A')}
Liều dùng: {item.get('lieu_dung', 'N/A')}
Công hiệu: {item.get('cong_hieu', 'N/A')}
Chú ý: {item.get('chu_y', 'Không có')}
Nguyên lý: {item.get('nguyen_ly', 'Không rõ')}
            """.strip()
            documents.append(doc)
        
        print(f"✅ Đã tải {len(documents)} tài liệu từ JSON")
        return documents
        
    except Exception as e:
        print(f"❌ Lỗi đọc JSON: {e}")
        return []

async def process_dongyi_documents(rag, dongyi_kg):
    """Xử lý tài liệu Đông y và tạo Knowledge Graph"""
    print("\n--- Xử lý Kiến thức Đông y và tạo Knowledge Graph ---")
    
    # Chọn nguồn dữ liệu
    print("\n📂 Chọn nguồn dữ liệu:")
    print("1. File văn bản (.txt) - phân cách bằng ---")
    print("2. File JSON")
    print("3. Sử dụng dữ liệu mẫu có sẵn")
    
    choice = input("Nhập lựa chọn (1/2/3): ").strip()
    
    if choice == "1":
        file_path = input("Nhập đường dẫn file .txt: ").strip()
        dongyi_documents = await load_documents_from_file(file_path)
    elif choice == "2":
        file_path = input("Nhập đường dẫn file .json: ").strip()
        dongyi_documents = await load_documents_from_json(file_path)
    else:
        # Dữ liệu mẫu mặc định
        dongyi_documents = [
            """
              Bột Cây Huệ Khô Nấu Cháo - Bài thuốc chữa ho ra máu

        Chữa trị: Ho ra máu (khi ho ít có những sợi máu nhỏ lẫn trong đờm, khi ho nhiều thì thường có cục máu, phần lớn do lao phổi, giãn khí quản).
        Thành phần: 30g bột cây huệ khô (tươi thì lượng gấp đôi), 100g gạo, đường phèn vừa đủ.
        Cách chế biến: Cho bột cây huệ, gạo và đường phèn vào nước, nấu thành cháo.
        Liều dùng: Ăn vào 2 bữa sáng chiều mỗi ngày.
        Đối tượng phù hợp: Thích hợp chữa trị phổi nóng ho ra máu.
        Công hiệu: Thanh nhiệt phổi, cầm máu, bổ khí.
        Chú ý: Người già tỳ vị hư hàn không được dùng kéo dài.
        Nguyên lý: Cây huệ có tính hàn, thanh nhiệt phổi, cầm máu.
            """,
            """
        Mộc Nhĩ Trắng Táo Tàu - Bài thuốc chữa lao ho ra máu

        Chữa trị: Ho ra máu do lao phổi, giãn khí quản.
        Thành phần: 10g mộc nhĩ trắng, 100g gạo tẻ, 5 quả táo tàu, đường phèn vừa đủ.
        Cách chế biến: Rửa sạch mộc nhĩ trắng, ngâm trong nước nóng 4 tiếng. Dùng gạo tẻ và táo cho nước vừa đủ, đun sôi. Sau đó cho mộc nhĩ và đường phèn vào nấu thành cháo.
        Liều dùng: Ăn cháo vào 2 buổi sáng, chiều mỗi ngày.
        Công hiệu: Chữa lao, ho ra máu, bổ phổi, nhuận tràng.
        Chú ý: Những người bị phong hàn cảm mạo tạm ngừng sử dụng bài thuốc này.
        Nguyên lý: Mộc nhĩ trắng có tính bình, nhuận phổi, cầm máu. Táo tàu bổ khí huyết.
        """
            # ... các bài thuốc khác ...
        ]
    
    if not dongyi_documents:
        print("❌ Không có tài liệu để xử lý!")
        return False
    
    try:
        # Xử lý từng tài liệu Đông y với LightRAG
        print(f"\n🔄 Bắt đầu xử lý {len(dongyi_documents)} tài liệu...")
        for i, doc in enumerate(dongyi_documents):
            print(f"   Đang xử lý tài liệu {i+1}/{len(dongyi_documents)}...")
            await rag.ainsert(doc.strip())
            print(f"   ✅ Đã xử lý tài liệu {i+1}")
            await asyncio.sleep(1)  # Đợi giữa các lần insert
        
        print("✅ Đã xử lý tất cả tài liệu với LightRAG")
        
        # Đợi để đảm bảo GraphML được tạo
        print("\n⏳ Đang đợi LightRAG tạo Knowledge Graph...")
        await asyncio.sleep(5)
        
        # Tìm và import GraphML file
        graphml_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
        
        if os.path.exists(graphml_file):
            print(f"\n📊 Đang import Knowledge Graph vào Neo4j...")
            dongyi_kg.import_from_graphml(graphml_file)
            
            # Hiển thị stats sau khi import
            print(f"\n✅ Hoàn tất import vào Neo4j!")
            dongyi_kg.get_stats()
            
        else:
            print(f"❌ Không tìm thấy GraphML file: {graphml_file}")
            print("Các file hiện có trong thư mục:")
            for f in os.listdir(WORKING_DIR):
                print(f"  - {f}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Lỗi xử lý tài liệu: {e}")
        traceback.print_exc()
        return False

async def main():
    """Hàm chính - CHẾ ĐỘ BỔ SUNG DỮ LIỆU"""
    
    # Kiểm tra cấu hình
    if not GEMINI_API_KEY:
        print("❌ Lỗi: Chưa cấu hình GEMINI_API_KEY")
        return
    
    rag = None
    dongyi_kg = None
    
    try:
        print(f"\n🚀 Bắt đầu Hệ thống Kiến thức Đông y + Neo4j")
        print(f"   Database: {NEO4J_DATABASE}")
        print(f"   Thư mục: {WORKING_DIR}")
        
        print("\n⚠️  CHẾ ĐỘ: BỔ SUNG DỮ LIỆU MỚI (KHÔNG XÓA DỮ LIỆU CŨ)")
        print("   ✓ LightRAG: Tự động merge dữ liệu mới vào vector DB")
        print("   ✓ Neo4j: Sử dụng MERGE để tránh trùng lặp entities/relationships")
        
        # Khởi tạo Neo4j
        print(f"\n--- Kết nối Neo4j Database: {NEO4J_DATABASE} ---")
        try:
            dongyi_kg = DongyiKnowledgeGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)
            
            # Test connection
            with dongyi_kg.driver.session(database=dongyi_kg.database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                print(f"✅ Kết nối Neo4j thành công")
            
            # Hiển thị dữ liệu hiện có
            print("\n📊 Dữ liệu hiện có TRƯỚC KHI bổ sung:")
            dongyi_kg.get_stats()
            
        except Exception as neo_error:
            print(f"❌ Lỗi kết nối Neo4j: {neo_error}")
            print("\n🔧 Kiểm tra lại:")
            print("   1. Neo4j có đang chạy không?")
            print("   2. Username/password có đúng không?") 
            print(f"   3. Database '{NEO4J_DATABASE}' có tồn tại không?")
            print(f"   4. Thử tạo database: CREATE DATABASE {NEO4J_DATABASE}")
            return
        
        # Khởi tạo LightRAG
        rag = await initialize_rag()
        if not rag:
            print("❌ Không thể khởi tạo LightRAG")
            return
        
        # Xử lý tài liệu
        success = await process_dongyi_documents(rag, dongyi_kg)
        if not success:
            print("❌ Không thể xử lý tài liệu")
            return
        
        # Thống kê cuối cùng
        print(f"\n" + "="*60)
        print(f"✅ HOÀN TẤT! Dữ liệu mới đã được bổ sung vào hệ thống")
        print(f"="*60)
        print(f"\n📂 Dữ liệu LightRAG: {WORKING_DIR}")
        print(f"💾 Neo4j Database: {NEO4J_DATABASE}")
        print(f"\n📊 Xem trong Neo4j Browser: http://localhost:7474")
        print(f"   Chọn database '{NEO4J_DATABASE}' và chạy query:")
        print(f"   MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 25")
        
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {e}")
        traceback.print_exc()
        
    finally:
        # Cleanup
        if rag:
            try:
                await rag.close_storages()
                print("\n✓ Đã đóng LightRAG")
            except:
                pass
        if dongyi_kg:
            dongyi_kg.close()
            print("✓ Đã đóng Neo4j connection")

if __name__ == "__main__":
    try:
        logging.basicConfig(
            level=logging.WARNING,
            format='%(levelname)s: %(message)s'
        )
        
        print("="*60)
        print("KHỞI ĐỘNG HỆ THỐNG KIẾN THỨC ĐÔNG Y")
        print("="*60)
        asyncio.run(main())
        print("\n🎉 Chương trình hoàn tất!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Chương trình bị dừng bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        traceback.print_exc()