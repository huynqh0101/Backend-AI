# Hệ thống LightRAG + Neo4j cho Kiến thức Đông y
# ------------------------------------------------
# Quản lý và truy vấn các bài thuốc Đông y truyền thống

import os
import asyncio
import logging
import numpy as np
import traceback
import xml.etree.ElementTree as ET
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
    
    def clear_database(self):
        """Dọn dẹp database Neo4j"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            print(f"Đã dọn dẹp Neo4j database: {self.database}")
    
    def create_entity(self, tx, entity_id, entity_type, description, source_id):
        """Tạo entity trong Neo4j (thuốc, bệnh, dược liệu...)"""
        query = """
        MERGE (e:Entity {id: $entity_id})
        SET e.type = $entity_type, 
            e.description = $description, 
            e.source_id = $source_id,
            e.displayName = $entity_id
        RETURN e
        """
        return tx.run(query, entity_id=entity_id, entity_type=entity_type, 
                     description=description, source_id=source_id)
    
    def create_relationship(self, tx, source_id, target_id, weight, description, keywords, source_doc):
        """Tạo relationship trong Neo4j"""
        query = """
        MATCH (source:Entity {id: $source_id})
        MATCH (target:Entity {id: $target_id})
        MERGE (source)-[r:CHUA_TRI]->(target)
        SET r.weight = $weight, 
            r.description = $description, 
            r.keywords = $keywords,
            r.source_doc = $source_doc
        RETURN r
        """
        return tx.run(query, source_id=source_id, target_id=target_id, 
                     weight=weight, description=description, 
                     keywords=keywords, source_doc=source_doc)
    
    def import_from_graphml(self, graphml_file):
        """Import GraphML file vào Neo4j"""
        try:
            print(f"Đang đọc GraphML file: {graphml_file}")
            tree = ET.parse(graphml_file)
            root = tree.getroot()
            
            # Định nghĩa namespace
            ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
            
            with self.driver.session(database=self.database) as session:
                # Import nodes (entities)
                nodes = root.findall('.//graphml:node', ns)
                print(f"Tìm thấy {len(nodes)} thành phần y học")
                
                entity_count = 0
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
                    
                    session.execute_write(self.create_entity, node_id, entity_type, description, "dongyi_import")
                    entity_count += 1
                
                print(f"Đã import {entity_count} thành phần y học")
                
                # Import edges (relationships)
                edges = root.findall('.//graphml:edge', ns)
                print(f"Tìm thấy {len(edges)} mối liên hệ chữa trị")
                
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
                
                print(f"Đã import {rel_count} mối liên hệ chữa trị")
                
        except Exception as e:
            print(f"Lỗi import GraphML: {e}")
            traceback.print_exc()
            raise e
    
    def get_stats(self):
        """Lấy thống kê database"""
        with self.driver.session(database=self.database) as session:
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r:CHUA_TRI]->() RETURN count(r) as count").single()["count"]
            
            print(f"Thống kê Kiến thức Đông y ({self.database}): {entity_count} thành phần, {rel_count} mối liên hệ chữa trị")
            return {"entities": entity_count, "relationships": rel_count}
    
    def query_dongyi(self, query_text):
        """Truy vấn kiến thức Đông y"""
        with self.driver.session(database=self.database) as session:
            # Tìm entities liên quan đến query
            cypher_query = """
            MATCH (e:Entity)
            WHERE toLower(e.description) CONTAINS toLower($query_text)
               OR toLower(e.id) CONTAINS toLower($query_text)
            RETURN e.id as entity_id, e.type as entity_type, e.description as description
            LIMIT 5
            """
            
            results = session.run(cypher_query, query_text=query_text)
            entities = [record.data() for record in results]
            
            if entities:
                print(f"Tìm thấy {len(entities)} bài thuốc/dược liệu liên quan:")
                for entity in entities:
                    print(f"  - {entity['entity_id']}: {entity['description'][:100]}...")
            
            return entities

# --- LightRAG Functions ---
async def gemini_llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
    """Custom LLM function sử dụng Gemini API cho Đông y"""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        if history_messages is None:
            history_messages = []

        # System prompt chuyên về Đông y
        dongyi_system_prompt = """Bạn là chuyên gia về y học cổ truyền Đông y. 
        Hãy trả lời chính xác về các bài thuốc, dược liệu, bệnh lý và phương pháp chữa trị theo Đông y."""
        
        combined_prompt = ""
        if system_prompt:
            combined_prompt += f"{dongyi_system_prompt}\n{system_prompt}\n"
        else:
            combined_prompt += f"{dongyi_system_prompt}\n"

        for msg in history_messages:
            combined_prompt += f"{msg['role']}: {msg['content']}\n"

        combined_prompt += f"user: {prompt}"

        response = client.models.generate_content(
            model="gemini-1.5-flash",
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
        
        print("LightRAG đã sẵn sàng cho Kiến thức Đông y")
        return rag
        
    except Exception as e:
        print(f"Lỗi khởi tạo LightRAG: {e}")
        traceback.print_exc()
        return None

async def cleanup_old_data():
    """Dọn dẹp các file dữ liệu cũ"""
    print("\n--- Dọn dẹp dữ liệu cũ ---")
    
    files_to_delete = [
        "graph_chunk_entity_relation.graphml",
        "kv_store_doc_status.json", 
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
    ]
    
    deleted_count = 0
    for file in files_to_delete:
        file_path = os.path.join(WORKING_DIR, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Đã xóa: {file}")
            deleted_count += 1
    
    print(f"Đã dọn dẹp {deleted_count} file cũ")

async def process_dongyi_documents(rag, dongyi_kg):
    """Xử lý tài liệu Đông y và tạo Knowledge Graph"""
    print("\n--- Xử lý Kiến thức Đông y và tạo Knowledge Graph ---")
    
    # Dữ liệu về các bài thuốc hạ sốt trong Đông y
    dongyi_documents = [
        """
        Cháo Lá Tre Thạch Cao - Bài thuốc hạ sốt số 1
        
        Chữa trị: Sốt cao (nhiệt độ cơ thể trên 39°C).
        Thành phần: 200g lá tre tươi rửa sạch, 100g thạch cao sống, 100g gạo tẻ.
        Cách chế biến: Cho lá tre và thạch cao vào 500ml nước sắc kỹ, lấy nước bỏ bã. 
        Cho gạo tẻ vào nước thuốc vừa sắc, nấu thành cháo.
        Liều dùng: Mỗi ngày ăn 2–3 lần.
        Công hiệu: Hạ hỏa, giải khát, giải phiền, bổ phổi.
        Chú ý: Khi nào cơn sốt lui thì ngừng uống thuốc.
        Nguyên lý: Lá tre có tính hàn, thanh nhiệt giải độc. Thạch cao thanh nhiệt tả hỏa.
        """,
        
        """
        Nước Giải Khát Ngũ Vị - Công thức hạ sốt số 2
        
        Chữa trị: Sốt cao.
        Thành phần: Nước quả lê, nước mã thầy, nước ngó sen, nước rễ lau sậy, nước mạch môn đông (hoặc nước mía).
        Cách chế biến: Lấy các loại nước trên với lượng bằng nhau, quấy đều, để lạnh.
        Liều dùng: Uống thay nước hàng ngày.
        Công hiệu: Thanh nhiệt, khỏi khát.
        Nguyên lý: Tất cả các thành phần đều có tính mát, sinh tân dịch, thanh nhiệt giải khát.
        Đặc điểm: Đây là phương thuốc tự nhiên, an toàn, không tác dụng phụ.
        """,
        
        """
        Rau Gan Chó với Đường Phèn - Bài thuốc dân gian số 3
        
        Chữa trị: Bệnh nhiệt và cảm cúm sốt cao.
        Thành phần: 30–60g rau gan chó, đường phèn vừa đủ.
        Cách chế biến: Sắc rau gan chó lấy nước, cho đường phèn vào.
        Liều dùng: Uống thay nước chè trong ngày.
        Công hiệu: Điều trị liên tục sẽ giúp làm lui cơn sốt.
        Nguyên lý: Rau gan chó có tính hàn, thanh nhiệt giải độc, kháng viêm.
        Ứng dụng: Thích hợp với trẻ em và người già.
        """,
        
        """
        Sừng Sơn Dương và Cây Câu Đằng - Bài thuốc quý số 4
        
        Chữa trị: Sốt cao.
        Thành phần: Sừng sơn dương 30g, cây câu đằng 6–10g.
        Cách chế biến: Cho nước vào sắc cùng nhau, lấy nước uống.
        Công hiệu: Thanh nhiệt, hết buồn phiền.
        Nguyên lý: Sừng sơn dương thanh nhiệt lương huyết, câu đằng thanh nhiệt giải độc.
        Đặc điểm: Là bài thuốc quý hiếm, công hiệu mạnh.
        Chú ý: Cần tìm nguồn sừng sơn dương chất lượng tốt.
        """,
        
        """
        Bột Sừng Trâu - Dược liệu quý số 5
        
        Chữa trị: Sốt cao.
        Thành phần: Bột sừng trâu.
        Cách chế biến: Sắc đặc bột sừng trâu lấy nước.
        Liều dùng: Uống mỗi ngày 1.5–3g, chia 2 lần trong ngày.
        Công hiệu: Thanh nhiệt, cắt cơn ho.
        Nguyên lý: Sừng trâu có tính hàn, thanh nhiệt lương huyết, giải độc.
        Ứng dụng: Đặc biệt hiệu quả với sốt cao kèm ho.
        """,
        
        """
        Bột Ngọc Trai với Sunfat Natri - Công thức đặc biệt số 6
        
        Chữa trị: Sốt cao.
        Thành phần: 0.3g bột ngọc trai nghiền nhỏ, 10g sunfat natri.
        Cách chế biến: Hãm bột ngọc trai vào 1 bát nước sôi, sau đó cho sunfat natri vào.
        Liều dùng: Uống hết trong 1 lần.
        Công hiệu: Thanh nhiệt, sinh huyết.
        Chú ý: Phụ nữ có thai không được dùng.
        Nguyên lý: Ngọc trai an thần định kinh, sunfat natri thanh nhiệt nhuận tràng.
        """,
        
        """
        Tằm Xác Ve và Ngân Hoa - Công thức phức hợp số 7
        
        Chữa trị: Sốt cao.
        Thành phần: 9g con tằm, 3g xác ve, 15g ngân hoa.
        Cách chế biến: 
        - Nghiền tằm và xác ve thành bột.
        - Sắc ngân hoa lấy nước.
        - Uống bột tằm, xác ve với nước ngân hoa.
        Biến thể: Nếu uống với nước sôi thì dùng 10g tằm, 12g xác ve.
        Công hiệu: Thanh nhiệt, mát phổi.
        Nguyên lý: Tằm và xác ve tức phong định kinh, ngân hoa thanh nhiệt giải độc.
        """
    ]
    
    try:
        # Xử lý từng tài liệu Đông y với LightRAG
        for i, doc in enumerate(dongyi_documents):
            print(f"Đang xử lý bài thuốc {i+1}/{len(dongyi_documents)}...")
            await rag.ainsert(doc.strip())
            print(f"Đã xử lý bài thuốc {i+1}")
            await asyncio.sleep(1)  # Đợi giữa các lần insert
        
        print("Đã xử lý tất cả bài thuốc Đông y với LightRAG")
        
        # Đợi để đảm bảo GraphML được tạo
        print("Đang đợi tạo Knowledge Graph...")
        await asyncio.sleep(5)
        
        # Tìm và import GraphML file
        graphml_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
        
        if os.path.exists(graphml_file):
            print(f"Đang import Knowledge Graph Đông y vào Neo4j...")
            dongyi_kg.import_from_graphml(graphml_file)
            
            # Hiển thị stats
            stats = dongyi_kg.get_stats()
            print(f"Knowledge Graph Đông y đã được tạo và lưu vào Neo4j!")
            
        else:
            print(f"Không tìm thấy GraphML file: {graphml_file}")
            print("Các file hiện có trong thư mục:")
            for f in os.listdir(WORKING_DIR):
                print(f"  - {f}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Lỗi xử lý tài liệu Đông y: {e}")
        traceback.print_exc()
        return False

async def test_dongyi_queries(rag, dongyi_kg):
    """Test truy vấn với cả LightRAG và Neo4j Knowledge Graph"""
    print("\n--- Test Truy vấn Kiến thức Đông y ---")
    
    queries = [
        "Thuốc nào chữa sốt cao hiệu quả nhất?",
        "Lá tre và thạch cao có tác dụng gì?",
        "Những dược liệu nào có thể hạ sốt?",
        "Bài thuốc nào phù hợp với trẻ em bị sốt?",
        "Ngọc trai có chữa được sốt cao không?",
        "Cách sử dụng rau gan chó chữa cảm cúm?",
        "Sừng trâu có tác dụng phụ gì không?"
    ]
    
    for query in queries:
        print(f"\nCâu hỏi: '{query}'")
        print("=" * 70)
        
        try:
            # 1. Truy vấn với LightRAG
            print("LightRAG (Kiến thức Đông y):")
            response = await rag.aquery(query, param=QueryParam(mode="naive"))
            print(f"   Trả lời: {response}")
            
            # 2. Truy vấn Knowledge Graph trong Neo4j
            print("Neo4j Knowledge Graph (Đông y):")
            with dongyi_kg.driver.session(database=dongyi_kg.database) as session:
                # Tìm entities liên quan đến query
                cypher_query = """
                MATCH (e:Entity)
                WHERE toLower(e.description) CONTAINS toLower($query_text)
                   OR toLower(e.id) CONTAINS toLower($query_text)
                RETURN e.id as entity_id, e.type as entity_type, e.description as description
                LIMIT 3
                """
                
                results = session.run(cypher_query, query_text=query)
                entities = [record.data() for record in results]
                
                if entities:
                    print(f"   Tìm thấy {len(entities)} bài thuốc/dược liệu liên quan:")
                    for entity in entities:
                        print(f"   - {entity['entity_id']}: {entity['description'][:150]}...")
                else:
                    print("   Không tìm thấy bài thuốc liên quan trong Knowledge Graph")
            
            print("-" * 70)
            
        except Exception as e:
            print(f"Lỗi khi truy vấn '{query}': {e}")

async def main():
    """Hàm chính"""
    
    # Kiểm tra cấu hình
    if not GEMINI_API_KEY:
        print("Lỗi: Chưa cấu hình GEMINI_API_KEY")
        return
    
    rag = None
    dongyi_kg = None
    
    try:
        print(f"\nBắt đầu Hệ thống Kiến thức Đông y + Neo4j (Database: {NEO4J_DATABASE})...")
        
        # Dọn dẹp dữ liệu cũ
        await cleanup_old_data()
        
        # Khởi tạo Neo4j cho Đông y
        print(f"\n--- Kết nối Neo4j Database Đông y: {NEO4J_DATABASE} ---")
        try:
            dongyi_kg = DongyiKnowledgeGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)
            
            # Test connection
            with dongyi_kg.driver.session(database=dongyi_kg.database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                print(f"Kết nối Neo4j thành công: {test_value}")
            
            dongyi_kg.clear_database()
            
        except Exception as neo_error:
            print(f"Lỗi kết nối Neo4j: {neo_error}")
            print("Kiểm tra lại:")
            print("1. Neo4j có đang chạy không?")
            print("2. Username/password có đúng không?") 
            print(f"3. Database '{NEO4J_DATABASE}' có tồn tại không?")
            print("4. Thử tạo database bằng lệnh: CREATE DATABASE dongyi")
            return
        
        # Khởi tạo LightRAG cho Đông y
        rag = await initialize_rag()
        if not rag:
            print("Không thể khởi tạo LightRAG")
            return
        
        # Test functions
        print("\n--- Test Functions ---")
        test_texts = ["Test embedding function for Traditional Medicine"]
        try:
            embeddings = await sentence_transformer_embedding_func(test_texts)
            print(f"Embedding OK: dimension {embeddings.shape}")
        except Exception as e:
            print(f"Lỗi embedding: {e}")
            return
        
        try:
            response = await gemini_llm_model_func("Cho tôi biết về công hiệu của lá tre trong Đông y.")
            print(f"LLM OK: {response[:100]}...")
        except Exception as e:
            print(f"Lỗi LLM: {e}")
            return
        
        # Xử lý tài liệu Đông y và tạo Knowledge Graph
        success = await process_dongyi_documents(rag, dongyi_kg)
        if not success:
            print("Không thể tạo Knowledge Graph Đông y")
            return
        
        # Test truy vấn
        await test_dongyi_queries(rag, dongyi_kg)
        
        print(f"\nHoàn tất! Hệ thống Kiến thức Đông y đã được lưu vào Neo4j database: {NEO4J_DATABASE}")
        print(f"Bạn có thể xem trong Neo4j Browser: http://localhost:7474")
        print(f"Nhớ chọn database '{NEO4J_DATABASE}' trong Neo4j Browser")
        print(f"Sử dụng Cypher query: MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 25")
        
    except Exception as e:
        print(f"Lỗi trong main: {e}")
        traceback.print_exc()
        
    finally:
        # Cleanup
        if rag:
            try:
                await rag.close_storages()
                print("Đã đóng LightRAG")
            except:
                pass
        if dongyi_kg:
            dongyi_kg.close()
            print("Đã đóng Neo4j connection")

if __name__ == "__main__":
    try:
        # Cấu hình logging đơn giản
        logging.basicConfig(
            level=logging.WARNING,
            format='%(levelname)s: %(message)s'
        )
        
        print("Khởi động Hệ thống Kiến thức Đông y + Neo4j...")
        asyncio.run(main())
        print("\nChương trình hoàn tất!")
        
    except KeyboardInterrupt:
        print("\nChương trình bị dừng bởi người dùng")
    except Exception as e:
        print(f"\nLỗi không mong muốn: {e}")
        traceback.print_exc()