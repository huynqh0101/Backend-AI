import os
import asyncio
import pandas as pd
import logging
import traceback
import xml.etree.ElementTree as ET
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from sentence_transformers import SentenceTransformer
from py2neo import Graph, Node, Relationship
import numpy as np
import nest_asyncio
import googletrans
from googletrans import Translator

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# ============= CẤU HÌNH =============
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "huy1552004"
NEO4J_DATABASE = "lightrag"

WORKING_DIR = "./lightrag_dongyi_neo4j"
CSV_FILE = "./data/data_translated.csv"
OLLAMA_BASE_URL = "http://localhost:11434"


# ============= OLLAMA LLM & EMBEDDING WRAPPER =============
async def ollama_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Wrapper để gọi Ollama API - Tương thích với LightRAG"""
    import httpx
    
    model = kwargs.get("model", "llama3.2:latest")
    
    # ✅ THÊM SYSTEM PROMPT ĐỂ ENFORCE FORMAT
    if not system_prompt:
        system_prompt = """Bạn là trợ lý AI chuyên trích xuất tri thức y học cổ truyền Việt Nam.
QUY TẮC QUAN TRỌNG:
1. Tất cả đầu ra PHẢI bằng tiếng Việt, tuyệt đối không dùng tiếng Anh.
2. Mô tả, nhãn, thuộc tính, keywords... đều phải là tiếng Việt.
3. Nếu không chắc, hãy trả về tiếng Việt đơn giản nhất.
4. Không thêm bình luận, giải thích hoặc dịch sang tiếng Anh.
5. Giữ nguyên thuật ngữ y học tiếng Việt."""
    
    # Tạo messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.0),
                        "num_ctx": kwargs.get("num_ctx", 8192),
                        "num_predict": 3072,  # ✅ TĂNG TỪ 2048 LÊN 3072
                        "top_k": 1,  # ✅ THÊM: Chỉ chọn token có xác suất cao nhất
                        "top_p": 0.1,  # ✅ THÊM: Nucleus sampling rất hẹp
                        "repeat_penalty": 1.1,  # ✅ THÊM: Tránh lặp lại
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["message"]["content"]
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                if response.status_code == 500:
                    error_msg += f"\nResponse: {response.text[:500]}"
                print(f"⚠️  {error_msg}")
                raise Exception(error_msg)
    except httpx.TimeoutException:
        print("⚠️  Ollama timeout - prompt quá dài hoặc model chậm")
        raise Exception("Ollama timeout")
    except Exception as e:
        print(f"⚠️  Ollama error: {e}")
        raise


async def sentence_transformer_embedding(texts: list[str]) -> np.ndarray:
    """Embedding function sử dụng SentenceTransformer"""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        print(f"❌ Lỗi embedding: {e}")
        raise


# ============= NEO4J KNOWLEDGE GRAPH CLASS =============
class DongyiKnowledgeGraph:
    """Quản lý Neo4j Knowledge Graph cho Đông y"""
    
    def __init__(self, uri, username, password, database="lightrag"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.graph = Graph(uri, auth=(username, password), name=database)
        print(f"✅ Kết nối Neo4j database: {database}")
    
    def clear_database(self):
        """Xóa toàn bộ dữ liệu cũ"""
        print("🗑️  Đang xóa dữ liệu cũ...")
        self.graph.run("MATCH (n) DETACH DELETE n")
        print("✅ Đã xóa dữ liệu cũ")
    
    def import_from_graphml(self, graphml_file):
        """Import GraphML file vào Neo4j - MERGE để tránh trùng"""
        try:
            print(f"\n📥 Đang import GraphML: {graphml_file}")
            tree = ET.parse(graphml_file)
            root = tree.getroot()
            
            ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
            
            # Import nodes
            nodes = root.findall('.//graphml:node', ns)
            print(f"   Tìm thấy {len(nodes)} nodes...")
            
            node_count = 0
            for node in nodes:
                node_id = node.get('id')
                entity_type = "Unknown"
                description = ""
                for data in node.findall('graphml:data', ns):
                    key = data.get('key')
                    if key in ['d1', 'entity_type']:
                        entity_type = data.text or "Unknown"
                    elif key in ['d2', 'description']:
                        description = ensure_vietnamese(data.text or "")
                # MERGE node để tránh trùng
                query = """
                MERGE (e:Entity {id: $node_id})
                ON CREATE SET 
                    e.type = $entity_type,
                    e.description = $description,
                    e.displayName = $node_id,
                    e.created_at = datetime()
                ON MATCH SET
                    e.type = $entity_type,
                    e.description = $description,
                    e.updated_at = datetime()
                """
                self.graph.run(query, node_id=node_id, entity_type=entity_type, 
                             description=description)
                node_count += 1
                
                if node_count % 100 == 0:
                    print(f"      Đã import {node_count}/{len(nodes)} nodes...")
            
            print(f"   ✅ Đã import {node_count} nodes")
            
            # Import edges
            edges = root.findall('.//graphml:edge', ns)
            print(f"   Tìm thấy {len(edges)} relationships...")
            
            rel_count = 0
            for edge in edges:
                source_id = edge.get('source')
                target_id = edge.get('target')
                weight = 1.0
                description = ""
                keywords = ""
                for data in edge.findall('graphml:data', ns):
                    key = data.get('key')
                    if key in ['d5', 'weight']:
                        try:
                            weight = float(data.text or 1.0)
                        except:
                            weight = 1.0
                    elif key in ['d6', 'description']:
                        description = ensure_vietnamese(data.text or "")
                    elif key in ['d7', 'keywords']:
                        keywords = data.text or ""
                
                # MERGE relationship
                query = """
                MATCH (source:Entity {id: $source_id})
                MATCH (target:Entity {id: $target_id})
                MERGE (source)-[r:RELATED]->(target)
                ON CREATE SET
                    r.weight = $weight,
                    r.description = $description,
                    r.keywords = $keywords,
                    r.created_at = datetime()
                ON MATCH SET
                    r.weight = $weight,
                    r.description = $description,
                    r.keywords = $keywords,
                    r.updated_at = datetime()
                """
                self.graph.run(query, source_id=source_id, target_id=target_id,
                             weight=weight, description=description, keywords=keywords)
                rel_count += 1
                
                if rel_count % 100 == 0:
                    print(f"      Đã import {rel_count}/{len(edges)} relationships...")
            
            print(f"   ✅ Đã import {rel_count} relationships")
            
        except Exception as e:
            print(f"❌ Lỗi import GraphML: {e}")
            traceback.print_exc()
    
    def get_stats(self):
        """Thống kê database"""
        entity_count = self.graph.run("MATCH (e:Entity) RETURN count(e) as count").evaluate()
        rel_count = self.graph.run("MATCH ()-[r:RELATED]->() RETURN count(r) as count").evaluate()
        
        print(f"\n📊 Thống kê Neo4j ({self.database}):")
        print(f"   - Entities: {entity_count}")
        print(f"   - Relationships: {rel_count}")
        return {"entities": entity_count, "relationships": rel_count}
    
    
def csv_to_documents(csv_path: str) -> str:
    """Chuyển CSV thành text documents"""
    print(f"\n📖 Đọc file CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines='skip', engine='python')
    
    print(f"✅ Đã đọc {len(df)} dòng dữ liệu")
    
    documents = []
    
    for idx, row in df.iterrows():
        chuong_so = row.get('chuong_so', '')
        tieu_de_chuong = row.get('tieu_de_chuong', '')
        ten_bai_thuoc = row.get('ten_bai_thuoc', '')
        chua_tri = row.get('chua_tri', '')
        lieu_luong_cach_dung = row.get('lieu_luong_cach_dung', '')
        cong_hieu = row.get('cong_hieu', '')
        chu_y = row.get('chu_y', '')
        doi_tuong_phu_hop = row.get('doi_tuong_phu_hop', '')
        
        if pd.isna(ten_bai_thuoc) or not ten_bai_thuoc:
            continue
        
        doc = f"""BÀI THUỐC: {ten_bai_thuoc}
Chữa trị: {chua_tri if pd.notna(chua_tri) else 'N/A'}
Liều lượng: {lieu_luong_cach_dung if pd.notna(lieu_luong_cach_dung) else 'N/A'}
Công hiệu: {cong_hieu if pd.notna(cong_hieu) else 'N/A'}
---
"""
        documents.append(doc)
        
        if idx < 3:
            print(f"\n📄 Document {idx + 1}:")
            print(doc[:150] + "...")
    
    print(f"\n✅ Đã tạo {len(documents)} documents")
    return "\n\n".join(documents)

async def initialize_lightrag():
    """Khởi tạo LightRAG với Ollama"""
    print("\n🚀 Khởi tạo LightRAG...")
    
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    try:
        rag = LightRAG(
            working_dir=WORKING_DIR,
            
            # LLM config - Ollama
            llm_model_func=ollama_model_complete,
            llm_model_name="llama3.2:latest",
            llm_model_max_async=1,
            llm_model_kwargs={
                "model": "llama3.2:latest",
                "temperature": 0.0,
                "num_ctx": 8192
            },
            
            # Embedding config - SentenceTransformer
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=512,
                func=sentence_transformer_embedding,
            ),
            
            # Graph config
            chunk_token_size=600,  # ✅ GIẢM TỪ 800 XUỐNG 600 (chunks nhỏ hơn = ít lỗi hơn)
            chunk_overlap_token_size=50,
        )
        
        print("   Đang khởi tạo storages...")
        await rag.initialize_storages()
        
        print("   Đang khởi tạo pipeline status...")
        await initialize_pipeline_status()
        
        print("✅ LightRAG đã sẵn sàng")
        return rag
        
    except Exception as e:
        print(f"❌ Lỗi khởi tạo LightRAG: {e}")
        traceback.print_exc()
        return None


# ============= BƯỚC 3: INSERT DỮ LIỆU =============
async def build_knowledge_graph(rag: LightRAG, documents: str):
    """Insert documents vào LightRAG"""
    print("\n📥 Bắt đầu insert dữ liệu vào LightRAG...")
    print(f"   Tổng độ dài: {len(documents)} ký tự")
    
    # ✅ GIẢM CHUNK SIZE ĐỂ TRÁNH QUÁ TẢI
    max_chunk_size = 20000  # Giảm từ 50000 xuống 20000
    chunks = []
    current_chunk = ""
    
    for doc in documents.split("\n\n"):
        if len(current_chunk) + len(doc) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = doc
        else:
            current_chunk += "\n\n" + doc
    
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"   Chia thành {len(chunks)} chunks")
    
    # Insert từng chunk
    for i, chunk in enumerate(chunks, 1):
        print(f"   📥 Đang insert chunk {i}/{len(chunks)}...")
        try:
            await rag.ainsert(chunk)
            print(f"   ✅ Chunk {i} hoàn tất")
            await asyncio.sleep(3)  # ✅ TĂNG DELAY
        except Exception as e:
            print(f"   ⚠️  Lỗi chunk {i}: {str(e)[:200]}")
            continue
    
    print("✅ Đã insert xong vào LightRAG!")


# ============= BƯỚC 4: TEST QUERY =============
async def test_query(rag: LightRAG):
    """Test query"""
    print("\n🔍 Test query...")
    
    test_queries = [
        "Bài thuốc nào chữa sốt cao?",
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        try:
            result = await rag.aquery(
                query,
                param=QueryParam(mode="naive", only_need_context=False, top_k=3)  # ✅ DÙNG "naive" MODE
            )
            print(f"📝 Kết quả:\n{result[:500]}...")
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            traceback.print_exc()


# ============= MAIN =============
async def main():
    print("="*70)
    print("🏥 LIGHTRAG + OLLAMA + NEO4J - ĐÔNG Y KNOWLEDGE GRAPH")
    print("="*70)
    
    dongyi_kg = None
    rag = None
    
    try:
        # Bước 1: Đọc CSV
        documents = csv_to_documents(CSV_FILE)
        
        # Bước 2: Khởi tạo LightRAG
        rag = await initialize_lightrag()
        if not rag:
            return
        
        # Bước 3: Build Knowledge Graph
        await build_knowledge_graph(rag, documents)
        
        # Bước 4: Test query
        await test_query(rag)
        
        # Bước 5: Import vào Neo4j
        print("\n" + "="*70)
        print("📤 IMPORT VÀO NEO4J")
        print("="*70)
        
        graphml_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
        
        if os.path.exists(graphml_file):
            dongyi_kg = DongyiKnowledgeGraph(NEO4J_URI, NEO4J_USERNAME, 
                                            NEO4J_PASSWORD, NEO4J_DATABASE)
            
            # Hỏi người dùng có muốn xóa dữ liệu cũ không
            choice = input("\n⚠️  Xóa dữ liệu cũ trong Neo4j? (y/n): ").strip().lower()
            if choice == 'y':
                dongyi_kg.clear_database()
            
            # Import GraphML
            dongyi_kg.import_from_graphml(graphml_file)
            
            # Hiển thị stats
            dongyi_kg.get_stats()
            
            print(f"\n📊 Xem trong Neo4j Browser: http://localhost:7474")
            print(f"   :use {NEO4J_DATABASE}")
            print(f"   MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 25")
            
        else:
            print(f"⚠️  Không tìm thấy GraphML: {graphml_file}")
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        traceback.print_exc()
    
    finally:
        if rag:
            try:
                await rag.close_storages()
                print("\n✓ Đã đóng LightRAG")
            except:
                pass
    
    print("\n" + "="*70)
    print("✅ HOÀN TẤT!")
    print("="*70)


if __name__ == "__main__":
    # Kiểm tra dependencies
    print("📦 Kiểm tra dependencies...")
    try:
        import httpx
        import sentence_transformers
        from py2neo import Graph
    except ImportError as e:
        print(f"⚠️  Thiếu package: {e}")
        print("Chạy: pip install httpx sentence-transformers py2neo")
        print(f"⚠️  Thiếu package: {e}")
        print("Chạy: pip install httpx sentence-transformers py2neo")
        exit(1)
    asyncio.run(main())