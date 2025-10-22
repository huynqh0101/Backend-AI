# Hệ thống Truy vấn Kiến thức Đông y - Chỉ Query
# ------------------------------------------------
# Truy vấn các bài thuốc Đông y mà không tạo Knowledge Graph mới

import os
import asyncio
import logging
import traceback
from google import genai
from google.genai import types
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# --- Cấu hình ---
print("--- Hệ thống Truy vấn Kiến thức Đông y ---")

# Gemini API
GEMINI_API_KEY = "AIzaSyDzXNvpMiLV9jbMUo-eZUcCxNXbXp2S4Ao"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Neo4j Configuration (nếu có sẵn)
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "huy1552004"
NEO4J_DATABASE = "dongyi"  # Database chuyên về Đông y

# Working Directory - sử dụng thư mục đã có
WORKING_DIR = "./dongyi_knowledge_graph"

print(f"Đã cấu hình hệ thống truy vấn Đông y")

# --- Neo4j Query Helper (Optional) ---
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
                # Debug: Kiểm tra tổng số entities
                count_result = session.run("MATCH (n) RETURN count(n) as total")
                total_entities = count_result.single()["total"]
                print(f"   Database có {total_entities} entities")
                
                if total_entities == 0:
                    print(f"   Database '{self.database}' trống!")
                    return []
                
                # Tìm kiếm thông minh hơn
                print(f"   Tìm kiếm: '{query_text}'")
                
                # Query 1: Tìm bài thuốc chính (recipes)
                recipe_query = """
                MATCH (n:Entity)
                WHERE (n.description IS NOT NULL AND n.description CONTAINS 'traditional' AND n.description CONTAINS 'medicine')
                   OR (n.description IS NOT NULL AND n.description CONTAINS 'recipe' AND n.description CONTAINS 'treat')
                   OR (n.id CONTAINS 'Cháo' OR n.id CONTAINS 'Nước' OR n.id CONTAINS 'Bột')
                   OR (n.description IS NOT NULL AND toLower(n.description) CONTAINS 'sốt' AND n.description CONTAINS 'treat')
                RETURN n.id as entity_id, 
                       labels(n) as entity_labels,
                       n.description as description,
                       n.displayName as displayName,
                       n.type as entity_type
                ORDER BY 
                    CASE 
                        WHEN n.description IS NULL THEN 0
                        ELSE size(n.description)
                    END DESC
                LIMIT 5
                """
                
                results = session.run(recipe_query)
                recipe_entities = []
                for record in results:
                    recipe_entities.append({
                        'entity_id': record.get('entity_id', 'N/A'),
                        'entity_type': record.get('entity_type', 'Recipe'),
                        'description': record.get('description', 'No description'),
                        'displayName': record.get('displayName', '')
                    })
                
                if recipe_entities:
                    print(f"   Tìm thấy {len(recipe_entities)} bài thuốc chính:")
                    return recipe_entities
                
                # Query 2: Tìm theo từ khóa trong description
                keyword_query = """
                MATCH (n:Entity)
                WHERE (n.description IS NOT NULL AND toLower(n.description) CONTAINS toLower($query_text))
                   OR (n.id IS NOT NULL AND toLower(n.id) CONTAINS toLower($query_text))
                   OR (n.displayName IS NOT NULL AND toLower(n.displayName) CONTAINS toLower($query_text))
                RETURN n.id as entity_id, 
                       labels(n) as entity_labels,
                       n.description as description,
                       n.displayName as displayName,
                       n.type as entity_type
                ORDER BY 
                    CASE 
                        WHEN n.id IS NOT NULL AND toLower(n.id) CONTAINS toLower($query_text) THEN 1
                        WHEN n.displayName IS NOT NULL AND toLower(n.displayName) CONTAINS toLower($query_text) THEN 2
                        ELSE 3
                    END,
                    CASE 
                        WHEN n.description IS NULL THEN 0
                        ELSE size(n.description)
                    END DESC
                LIMIT 5
                """
                
                results = session.run(keyword_query, query_text=query_text)
                entities = []
                for record in results:
                    entities.append({
                        'entity_id': record.get('entity_id', 'N/A'),
                        'entity_type': record.get('entity_type', str(record.get('entity_labels', []))),
                        'description': record.get('description', 'No description'),
                        'displayName': record.get('displayName', '')
                    })
                
                if entities:
                    print(f"   Tìm thấy {len(entities)} kết quả phù hợp")
                    return entities
                
                # Query 3: Tìm với từ khóa cụ thể về Đông y
                specific_keywords = {
                    "sốt": ["sốt", "fever", "high temperature"],
                    "thuốc": ["thuốc", "medicine", "remedy", "formula"],
                    "chữa": ["chữa", "treat", "cure", "therapy"],
                    "lá tre": ["lá tre", "bamboo", "leaves"],
                    "thạch cao": ["thạch cao", "gypsum"],
                }
                
                for main_keyword, variations in specific_keywords.items():
                    if main_keyword.lower() in query_text.lower():
                        for variation in variations:
                            specific_query = """
                            MATCH (n:Entity)
                            WHERE (n.description IS NOT NULL AND toLower(n.description) CONTAINS toLower($keyword))
                               OR (n.id IS NOT NULL AND toLower(n.id) CONTAINS toLower($keyword))
                            RETURN n.id as entity_id, 
                                   n.description as description,
                                   n.type as entity_type,
                                   n.displayName as displayName
                            ORDER BY 
                                CASE 
                                    WHEN n.description IS NULL THEN 0
                                    ELSE size(n.description)
                                END DESC
                            LIMIT 3
                            """
                            
                            results = session.run(specific_query, keyword=variation)
                            entities = []
                            for record in results:
                                entities.append({
                                    'entity_id': record.get('entity_id', 'N/A'),
                                    'entity_type': record.get('entity_type', 'Entity'),
                                    'description': record.get('description', 'No description'),
                                    'displayName': record.get('displayName', '')
                                })
                            
                            if entities:
                                print(f"   Tìm thấy {len(entities)} kết quả với '{variation}'")
                                return entities
                
                # Query 4: Fallback - lấy samples có description
                fallback_query = """
                MATCH (n:Entity)
                WHERE n.description IS NOT NULL
                RETURN n.id as entity_id, 
                       n.description as description,
                       n.type as entity_type,
                       n.displayName as displayName
                ORDER BY size(n.description) DESC
                LIMIT 3
                """
                
                results = session.run(fallback_query)
                entities = []
                for record in results:
                    entities.append({
                        'entity_id': record.get('entity_id', 'N/A'),
                        'entity_type': record.get('entity_type', 'Entity'),
                        'description': record.get('description', 'No description'),
                        'displayName': record.get('displayName', '')
                    })
                
                if entities:
                    print(f"   Không tìm thấy kết quả cụ thể, hiển thị {len(entities)} samples:")
                    return entities
                
                print("   Không tìm thấy kết quả nào")
                return []
                
        except Exception as e:
            print(f"Lỗi truy vấn Neo4j: {e}")
            return []
    
    def get_all_remedies(self):
        """Lấy tất cả bài thuốc trong database"""
        try:
            with self.driver.session(database=self.database) as session:
                # Query để tìm bài thuốc
                cypher_query = """
                MATCH (e:Entity)
                WHERE (e.description IS NOT NULL AND e.description CONTAINS 'traditional' AND e.description CONTAINS 'medicine')
                   OR (e.description IS NOT NULL AND e.description CONTAINS 'recipe' AND e.description CONTAINS 'treat')
                   OR (e.description IS NOT NULL AND e.description CONTAINS 'Vietnamese herbal')
                   OR (toLower(e.id) CONTAINS 'cháo' OR toLower(e.id) CONTAINS 'nước' 
                       OR toLower(e.id) CONTAINS 'bột' OR toLower(e.id) CONTAINS 'rau')
                RETURN e.id as entity_id, 
                       e.type as entity_type, 
                       e.description as description,
                       e.displayName as displayName
                ORDER BY 
                    CASE 
                        WHEN e.description IS NULL THEN 0
                        ELSE size(e.description)
                    END DESC
                LIMIT 10
                """
                
                results = session.run(cypher_query)
                entities = []
                for record in results:
                    entities.append({
                        'entity_id': record.get('entity_id', 'N/A'),
                        'entity_type': record.get('entity_type', 'Recipe'),
                        'description': record.get('description', 'No description'),
                        'displayName': record.get('displayName', '')
                    })
                
                return entities
        except Exception as e:
            print(f"Lỗi lấy danh sách bài thuốc: {e}")
            return []
    
    def debug_database(self):
        """Debug thông tin database"""
        try:
            with self.driver.session(database=self.database) as session:
                print(f"\nDEBUG DATABASE: {self.database}")
                
                # 1. Kiểm tra database có tồn tại không
                try:
                    count_result = session.run("MATCH (n) RETURN count(n) as total")
                    total = count_result.single()["total"]
                    print(f"   Database '{self.database}' accessible")
                    print(f"   Tổng số nodes: {total}")
                    
                    if total == 0:
                        print(f"   Database '{self.database}' TRỐNG!")
                        print(f"   Hãy chạy lightrag_dongyi.py để tạo dữ liệu")
                        return
                    
                    # 2. Kiểm tra labels
                    labels_query = """
                    MATCH (n)
                    RETURN DISTINCT labels(n) as labels, count(n) as count
                    ORDER BY count DESC
                    LIMIT 10
                    """
                    results = session.run(labels_query)
                    print("   Các loại nodes:")
                    for record in results:
                        print(f"      - {record['labels']}: {record['count']} nodes")
                    
                    # 3. Sample nodes
                    sample_query = """
                    MATCH (n)
                    RETURN n.id as id, labels(n) as labels, 
                           substring(coalesce(n.description, 'No description'), 0, 80) as desc
                    LIMIT 5
                    """
                    results = session.run(sample_query)
                    print("   Sample nodes:")
                    for record in results:
                        print(f"      - {record['id']}: {record['desc']}...")
                        
                except Exception as db_error:
                    print(f"   Database '{self.database}' không tồn tại hoặc không access được")
                    print(f"   Thử database khác: 'neo4j', 'lightrag'")
                    
        except Exception as e:
            print(f"Lỗi debug database: {e}")

    def check_all_databases(self):
        """Kiểm tra tất cả databases có sẵn"""
        try:
            with self.driver.session() as session:
                # Lệnh SHOW DATABASES (Neo4j 4.0+)
                try:
                    result = session.run("SHOW DATABASES")
                    print("\nTẤT CẢ DATABASES:")
                    for record in result:
                        db_name = record.get('name')
                        status = record.get('currentStatus', 'unknown')
                        print(f"   - {db_name}: {status}")
                        
                        # Check từng database có data không
                        if status == 'online':
                            try:
                                with self.driver.session(database=db_name) as db_session:
                                    count_result = db_session.run("MATCH (n) RETURN count(n) as total")
                                    total = count_result.single()["total"]
                                    print(f"     {total} nodes")
                            except:
                                print(f"     Không access được")
                                
                except Exception as show_error:
                    print(f"   Không thể SHOW DATABASES: {show_error}")
                    print(f"   Thử manual check database: neo4j, lightrag, dongyi")
                    
        except Exception as e:
            print(f"Lỗi check databases: {e}")

# --- LightRAG Functions ---
async def gemini_llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
    """Custom LLM function sử dụng Gemini API cho Đông y"""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        if history_messages is None:
            history_messages = []

        # System prompt chuyên về Đông y
        dongyi_system_prompt = """Bạn là chuyên gia về y học cổ truyền Đông y. 
        Hãy trả lời chính xác và chi tiết về các bài thuốc, dược liệu, bệnh lý và phương pháp chữa trị theo Đông y.
        Khi trả lời, hãy bao gồm: thành phần, cách chế biến, liều dùng, công hiệu, và chú ý quan trọng."""
        
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
            config=types.GenerateContentConfig(max_output_tokens=1500, temperature=0.1),
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

async def initialize_existing_rag():
    """Khởi tạo LightRAG với dữ liệu đã có"""
    print("\n--- Khởi tạo LightRAG với dữ liệu Đông y có sẵn ---")
    
    # Kiểm tra thư mục có tồn tại không
    if not os.path.exists(WORKING_DIR):
        print(f"Không tìm thấy thư mục: {WORKING_DIR}")
        print("Vui lòng chạy chương trình tạo Knowledge Graph trước!")
        return None
    
    # Kiểm tra file cần thiết
    required_files = [
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json", 
        "vdb_chunks.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(WORKING_DIR, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"Thiếu các file cần thiết: {missing_files}")
        print("Vui lòng chạy chương trình tạo Knowledge Graph trước!")
        return None
    
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
        
        # Khởi tạo storages để load dữ liệu có sẵn
        print("Đang khởi tạo storages...")
        await rag.initialize_storages()
        
        print("LightRAG đã sẵn sàng với dữ liệu Đông y có sẵn")
        return rag
        
    except Exception as e:
        print(f"Lỗi khởi tạo LightRAG: {e}")
        traceback.print_exc()
        return None

async def interactive_dongyi_query():
    """Chế độ truy vấn tương tác"""
    print("\n=== CHƯƠNG TRÌNH TRUY VẤN ĐÔNG Y TƯƠNG TÁC ===")
    print("Nhập 'exit' để thoát, 'help' để xem hướng dẫn")
    
    # Khởi tạo LightRAG
    rag = await initialize_existing_rag()
    if not rag:
        return
    
    # Khởi tạo Neo4j helper (optional)
    neo4j_helper = None
    try:
        neo4j_helper = DongyiQueryHelper(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)
        
        # Test connection
        with neo4j_helper.driver.session(database=neo4j_helper.database) as session:
            session.run("RETURN 1")
        print("Kết nối Neo4j thành công")
        
    except Exception as e:
        print(f"Không kết nối được Neo4j: {e}")
        print("   Chỉ sử dụng LightRAG")
        neo4j_helper = None
    
    try:
        while True:
            try:
                print("\n" + "="*50)
                user_query = input("Nhập câu hỏi về Đông y: ").strip()
                
                if user_query.lower() == 'exit':
                    print("Tạm biệt!")
                    break
                elif user_query.lower() == 'help':
                    print_help()
                    continue
                elif user_query.lower() == 'examples':
                    print_examples()
                    continue
                elif user_query.lower() == 'list':
                    if neo4j_helper:
                        show_all_remedies(neo4j_helper)
                    else:
                        print("Cần kết nối Neo4j để xem danh sách")
                    continue
                elif user_query.lower() == 'debug':
                    if neo4j_helper:
                        neo4j_helper.debug_database()
                    else:
                        print("Cần kết nối Neo4j để debug")
                    continue
                elif user_query.lower() == 'databases':
                    if neo4j_helper:
                        neo4j_helper.check_all_databases()
                    else:
                        print("Cần kết nối Neo4j để check databases")
                    continue
                elif not user_query:
                    print("Vui lòng nhập câu hỏi!")
                    continue
                
                print(f"\nĐang tìm kiếm thông tin về: '{user_query}'...")
                
                # 1. Truy vấn với LightRAG
                print(f"\n**LightRAG - Kiến thức Đông y:**")
                try:
                    response = await rag.aquery(user_query, param=QueryParam(mode="naive"))
                    print(f"{response}")
                except Exception as e:
                    print(f"Lỗi LightRAG: {e}")
                
                # 2. Truy vấn Neo4j Knowledge Graph (nếu có)
                if neo4j_helper:
                    print(f"\n**Neo4j Knowledge Graph:**")
                    entities = neo4j_helper.query_dongyi_kg(user_query)
                    
                    if entities:
                        print(f"\n   Kết quả từ Knowledge Graph:")
                        for i, entity in enumerate(entities, 1):
                            print(f"\n   {i}. {entity['entity_id']}")
                            print(f"   Loại: {entity.get('entity_type', 'N/A')}")
                            
                            # Hiển thị description
                            desc = entity.get('description', 'No description')
                            if len(desc) > 300:
                                print(f"   Mô tả: {desc[:300]}...")
                                print(f"   [Mô tả đã rút gọn - có thể chứa nhiều thông tin hơn]")
                            else:
                                print(f"   Mô tả: {desc}")
                                
                            # Thêm thông tin về displayName nếu khác với id
                            display_name = entity.get('displayName', '')
                            if display_name and display_name != entity['entity_id']:
                                print(f"   Tên hiển thị: {display_name}")
                    else:
                        print("   Không tìm thấy thông tin liên quan trong Knowledge Graph")
                        print("   Thử câu hỏi cụ thể hơn hoặc sử dụng từ khóa: 'sốt', 'thuốc', 'lá tre'")
                
            except KeyboardInterrupt:
                print("\nTạm biệt!")
                break
            except Exception as e:
                print(f"Lỗi: {e}")
        
    finally:
        # Cleanup
        print("\nĐang dọn dẹp...")
        if rag:
            try:
                await rag.close_storages()
                print("Đã đóng LightRAG")
            except:
                pass
        if neo4j_helper:
            neo4j_helper.close()
            print("Đã đóng Neo4j")

def print_help():
    """In hướng dẫn sử dụng"""
    print("\n**HƯỚNG DẪN SỬ DỤNG:**")
    print("• Nhập câu hỏi về Đông y để tìm kiếm")
    print("• 'help' - Xem hướng dẫn này")
    print("• 'examples' - Xem ví dụ câu hỏi")
    print("• 'list' - Xem danh sách bài thuốc (cần Neo4j)")
    print("• 'exit' - Thoát chương trình")
    
    print("\n**MẸO:**")
    print("• Hỏi cụ thể: 'Cách chữa sốt cao bằng lá tre'")
    print("• Hỏi về tác dụng: 'Thạch cao có công hiệu gì'")
    print("• Hỏi về liều dùng: 'Liều dùng rau gan chó'")

def print_examples():
    """In ví dụ câu hỏi"""
    print("\n**VÍ DỤ CÂU HỎI:**")
    examples = [
        "Thuốc nào chữa sốt cao hiệu quả nhất?",
        "Lá tre và thạch cao có tác dụng gì?",
        "Cách sử dụng rau gan chó chữa cảm cúm?",
        "Bài thuốc nào phù hợp với trẻ em bị sốt?",
        "Ngọc trai có chữa được sốt cao không?",
        "Sừng trâu có tác dụng phụ gì không?",
        "Cách pha chế nước giải khát ngũ vị?",
        "Thành phần của cháo lá tre thạch cao?",
        "Liều lượng sử dụng bột sừng trâu?",
        "Chú ý khi dùng bột ngọc trai?"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"   {i:2d}. {example}")

def show_all_remedies(neo4j_helper):
    """Hiển thị tất cả bài thuốc"""
    print("\n**DANH SÁCH BÀI THUỐC TRONG KNOWLEDGE GRAPH:**")
    
    remedies = neo4j_helper.get_all_remedies()
    if remedies:
        for i, remedy in enumerate(remedies, 1):
            print(f"\n{i}. {remedy['entity_id']}")
            print(f"   Loại: {remedy.get('entity_type', 'N/A')}")
            print(f"   Mô tả: {remedy['description'][:300]}...")
    else:
        print("Không tìm thấy bài thuốc nào")

async def batch_query_test():
    """Test một loạt câu hỏi cố định"""
    print("\n=== TEST BATCH QUERIES ===")
    
    # Khởi tạo LightRAG
    rag = await initialize_existing_rag()
    if not rag:
        return
    
    test_queries = [
        "Thuốc nào chữa sốt cao hiệu quả nhất?",
        "Lá tre và thạch cao có tác dụng gì?",
        "Cách sử dụng rau gan chó chữa cảm cúm?",
        "Bài thuốc nào phù hợp với trẻ em bị sốt?",
        "Sừng trâu có tác dụng phụ gì không?"
    ]
    
    try:
        for i, query in enumerate(test_queries, 1):
            print(f"\n**Test {i}/5:** {query}")
            print("-" * 60)
            
            try:
                response = await rag.aquery(query, param=QueryParam(mode="naive"))
                print(f"**Trả lời:** {response}")
            except Exception as e:
                print(f"Lỗi: {e}")
            
            print()
    
    finally:
        # Cleanup
        print("\nĐang dọn dẹp...")
        try:
            await rag.close_storages()
            print("Đã đóng LightRAG")
        except:
            pass

async def main():
    """Hàm chính"""
    
    # Kiểm tra cấu hình
    if not GEMINI_API_KEY:
        print("Lỗi: Chưa cấu hình GEMINI_API_KEY")
        return
    
    print("\n**HỆ THỐNG TRUY VẤN KIẾN THỨC ĐÔNG Y**")
    print("=" * 50)
    
    # Chọn chế độ
    print("\nChọn chế độ hoạt động:")
    print("1. Truy vấn tương tác")
    print("2. Test batch queries")
    
    try:
        choice = input("\nNhập lựa chọn (1/2): ").strip()
        
        if choice == "1":
            await interactive_dongyi_query()
        elif choice == "2":
            await batch_query_test()
        else:
            print("Lựa chọn không hợp lệ!")
            
    except KeyboardInterrupt:
        print("\nTạm biệt!")
    except Exception as e:
        print(f"Lỗi: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        # Cấu hình logging đơn giản
        logging.basicConfig(
            level=logging.WARNING,
            format='%(levelname)s: %(message)s'
        )
        
        print("Khởi động Hệ thống Truy vấn Đông y...")
        asyncio.run(main())
        print("\nChương trình hoàn tất!")
        
    except KeyboardInterrupt:
        print("\nChương trình bị dừng bởi người dùng")
    except Exception as e:
        print(f"\nLỗi không mong muốn: {e}")
        traceback.print_exc()