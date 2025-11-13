import asyncio
from py2neo import Graph
import re
from unidecode import unidecode

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "huy1552004"
NEO4J_DATABASE = "lightrag"

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
    """Chuẩn hóa text: bỏ dấu, chuyển thường, bỏ ký tự đặc biệt"""
    text = unidecode(text.lower())
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def query_neo4j(graph: Graph, user_query: str, top_k: int = 5):
    """Truy vấn Neo4j để tìm entities và relations liên quan"""
    # Chuẩn hóa query
    normalized_query = normalize_text(user_query)
    keywords = normalized_query.split()
    
    # Tìm entities - tìm theo cả bản gốc và bản chuẩn hóa
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
    
    # Tìm relationships
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
    OR toLower(e1.displayName) CONTAINS toLower($original_query)
    OR toLower(e2.displayName) CONTAINS toLower($original_query)
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

def build_context(entities, relations):
    """Xây dựng context từ kết quả Neo4j"""
    context = "## Thông tin từ cơ sở tri thức Đông y:\n\n"
    
    if entities:
        context += "### Các khái niệm liên quan:\n"
        for e in entities:
            context += f"- **{e['name']}** ({e['type']}): {e['description']}\n"
        context += "\n"
    
    if relations:
        context += "### Các mối quan hệ:\n"
        for r in relations:
            context += f"- {r['source']} → {r['target']}: {r['relation']}\n"
        context += "\n"
    
    return context

async def main():
    print("🔍 Truy vấn Neo4j DB Đông y")
    graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), name=NEO4J_DATABASE)
    
    while True:
        user_query = input("\nNhập câu hỏi Đông y (hoặc 'exit' để thoát): ").strip()
        if user_query.lower() == "exit":
            break
        
        try:
            # Truy vấn Neo4j
            entities, relations = query_neo4j(graph, user_query, top_k=5)
            
            if not entities and not relations:
                print("\nKhông tìm thấy thông tin liên quan trong cơ sở tri thức.")
                continue
            
            # Xây dựng context
            context = build_context(entities, relations)
            
            # Dùng Ollama để sinh câu trả lời
            prompt = f"""Dựa vào thông tin sau từ cơ sở tri thức Đông y, hãy trả lời câu hỏi của người dùng.

{context}

Câu hỏi: {user_query}

Hãy trả lời một cách rõ ràng, chi tiết và bằng tiếng Việt."""
            
            
            print("\n🤖 Đang sinh câu trả lời...")
            answer = await ollama_model_complete(prompt, model="llama3.2:latest", temperature=0.0)
            
            print("\n✅ Câu trả lời:")
            print(answer)
            
        except Exception as e:
            print(f"Lỗi truy vấn: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())