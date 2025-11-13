import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer
from py2neo import Graph
import re
from unidecode import unidecode

WORKING_DIR = "./lightrag_dongyi_neo4j"
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "huy1552004"
NEO4J_DATABASE = "lightrag"

async def sentence_transformer_embedding(texts: list[str]):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, convert_to_numpy=True)

async def ollama_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    import httpx
    model = kwargs.get("model", "llama3.2:latest")
    if not system_prompt:
        system_prompt = "Bạn là trợ lý AI Đông y, trả lời bằng tiếng Việt."
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.0),
                    "num_ctx": kwargs.get("num_ctx", 8192),
                    "num_predict": 3072,
                    "top_k": 1,
                    "top_p": 0.1,
                    "repeat_penalty": 1.1,
                }
            }
        )
        result = response.json()
        return result["message"]["content"]

def normalize_text(text: str) -> str:
    """Chuẩn hóa text: bỏ dấu, chuyển thường"""
    text = unidecode(text.lower())
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def query_neo4j(graph: Graph, user_query: str, top_k: int = 5):
    """Truy vấn Neo4j để lấy entities và relations"""
    normalized_query = normalize_text(user_query)
    keywords = normalized_query.split()
    
    entity_query = """
    MATCH (e:Entity)
    WHERE ANY(keyword IN $keywords WHERE 
        toLower(e.description) CONTAINS keyword OR 
        toLower(e.displayName) CONTAINS keyword
    )
    OR toLower(e.description) CONTAINS toLower($original_query)
    OR toLower(e.displayName) CONTAINS toLower($original_query)
    RETURN e.displayName as name, e.description as description, e.type as type
    LIMIT $limit
    """
    entities = graph.run(entity_query, 
                        keywords=keywords, 
                        original_query=user_query,
                        limit=top_k).data()
    
    rel_query = """
    MATCH (e1:Entity)-[r:RELATED]->(e2:Entity)
    WHERE ANY(keyword IN $keywords WHERE 
        toLower(e1.description) CONTAINS keyword OR
        toLower(e2.description) CONTAINS keyword OR
        toLower(r.description) CONTAINS keyword OR
        toLower(e1.displayName) CONTAINS keyword OR
        toLower(e2.displayName) CONTAINS keyword
    )
    OR toLower(e1.description) CONTAINS toLower($original_query)
    OR toLower(e2.description) CONTAINS toLower($original_query)
    OR toLower(r.description) CONTAINS toLower($original_query)
    RETURN e1.displayName as source, 
           r.description as relation, 
           e2.displayName as target,
           r.weight as weight
    ORDER BY r.weight DESC
    LIMIT $limit
    """
    relations = graph.run(rel_query, 
                         keywords=keywords,
                         original_query=user_query,
                         limit=top_k).data()
    
    return entities, relations

def build_neo4j_context(entities, relations):
    """Xây dựng context từ Neo4j"""
    context = "## Thông tin từ Knowledge Graph (Neo4j):\n\n"
    
    if entities:
        context += "### Các khái niệm liên quan:\n"
        for e in entities:
            # Rút gọn description để tránh quá dài
            desc = e['description'][:300] + "..." if len(e['description']) > 300 else e['description']
            context += f"- **{e['name']}** ({e['type']}): {desc}\n"
        context += "\n"
    
    if relations:
        context += "### Các mối quan hệ:\n"
        for r in relations:
            rel_desc = r['relation'][:200] + "..." if len(r['relation']) > 200 else r['relation']
            context += f"- {r['source']} → {r['target']}: {rel_desc}\n"
        context += "\n"
    
    return context

async def main():
    print("🔍 Truy vấn Hybrid: LightRAG Vector + Neo4j Graph")
    
    # Khởi tạo LightRAG
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="llama3.2:latest",
        llm_model_kwargs={
            "model": "llama3.2:latest",
            "temperature": 0.0,
            "num_ctx": 8192
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=512,
            func=sentence_transformer_embedding,
        ),
        chunk_token_size=600,
        chunk_overlap_token_size=50,
    )
    await rag.initialize_storages()
    
    # Kết nối Neo4j
    graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), name=NEO4J_DATABASE)
    
    while True:
        user_query = input("\nNhập câu hỏi Đông y (hoặc 'exit' để thoát): ").strip()
        if user_query.lower() == "exit":
            break
        
        try:
            print("\n📊 Đang tìm kiếm...")
            
            # 1. LightRAG vector search để lấy context từ chunks
            lightrag_context = await rag.aquery(
                user_query,
                param=QueryParam(
                    mode="naive",
                    only_need_context=True,
                    top_k=3
                )
            )
            
            # 2. Neo4j graph query để lấy entities và relations
            entities, relations = query_neo4j(graph, user_query, top_k=5)
            neo4j_context = build_neo4j_context(entities, relations) if (entities or relations) else ""
            
            # 3. Kết hợp cả 2 nguồn context
            combined_context = f"""## Context từ Vector Search (LightRAG):
{lightrag_context}

{neo4j_context}
"""
            
            print(f"\n📚 Tìm thấy: {len(entities)} entities, {len(relations)} relations")
            
            # 4. Prompt tối ưu cho LLM
            prompt = f"""Bạn là chuyên gia Y học cổ truyền Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên thông tin được cung cấp.

## HƯỚNG DẪN:
1. Đọc kỹ toàn bộ thông tin từ cơ sở tri thức bên dưới
2. Tổng hợp và phân tích thông tin liên quan đến câu hỏi
3. Trả lời bằng tiếng Việt, rõ ràng, đầy đủ và chính xác
4. Nếu có nhiều bài thuốc/phương pháp, hãy liệt kê từng mục với cấu trúc:
   - Tên bài thuốc/phương pháp
   - Công dụng/chữa bệnh gì
   - Thành phần (nếu có)
   - Cách dùng/liều lượng (nếu có)
5. Chỉ sử dụng thông tin có trong context, KHÔNG bịa đặt
6. Nếu không đủ thông tin, hãy nói rõ phần nào thiếu

## THÔNG TIN TỪ CƠ SỞ TRI THỨC:
{combined_context}

## CÂU HỎI CỦA NGƯỜI DÙNG:
{user_query}

## CÂU TRẢ LỜI CỦA BẠN (chỉ bằng tiếng Việt):"""
            
            print("\n🤖 Đang sinh câu trả lời...")
            answer = await ollama_model_complete(prompt, model="llama3.2:latest", temperature=0.0)
            
            print("\n✅ Câu trả lời:")
            print(answer)
            
        except Exception as e:
            print(f"Lỗi truy vấn: {e}")
            import traceback
            traceback.print_exc()
    
    await rag.close_storages()

if __name__ == "__main__":
    asyncio.run(main())