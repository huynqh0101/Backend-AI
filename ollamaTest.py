import os
import logging
import requests
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from py2neo import Graph, Node, Relationship
import xml.etree.ElementTree as ET

# Set logging level
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# Custom Ollama functions (vì lightrag.llm không export)
async def ollama_model_complete(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    **kwargs
):
    host = kwargs.get("host", "http://localhost:11434")
    model = kwargs.get("model_name", "qwen2:7b")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": kwargs.get("options", {})
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        logging.error(f"Ollama completion error: {e}")
        raise

async def ollama_embedding(texts):
    """Wrapper cho Ollama embedding API - ASYNC"""
    host = "http://localhost:11434"
    embed_model = "nomic-embed-text"
    
    embeddings = []
    try:
        for text in texts:
            response = requests.post(
                f"{host}/api/embeddings",
                json={
                    "model": embed_model,
                    "prompt": text
                },
                timeout=60
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings
    except Exception as e:
        logging.error(f"Ollama embedding error: {e}")
        raise

documents = [
    "Trí tuệ nhân tạo (AI) là lĩnh vực khoa học máy tính nghiên cứu cách tạo ra các hệ thống có thể mô phỏng trí thông minh của con người.",
    "Máy học là một công nghệ cốt lõi của AI, cho phép máy tính học hỏi và cải thiện từ dữ liệu mà không cần lập trình rõ ràng.",
    "Học sâu là một nhánh của máy học sử dụng mạng nơ-ron nhiều tầng để giải quyết các vấn đề phức tạp như nhận diện hình ảnh và xử lý ngôn ngữ tự nhiên."
]

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create working directory
        WORKING_DIR = "./my_rag_project"
        os.makedirs(WORKING_DIR, exist_ok=True)
        
        # Initialize LightRAG with Ollama model
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=ollama_model_complete,
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=ollama_embedding,
            ),
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
        )
        
        # ✅ QUAN TRỌNG: Khởi tạo storages TRƯỚC KHI insert
        print("Initializing storages...")
        await rag.initialize_storages()
        await initialize_pipeline_status()
        
        # Insert documents
        print("Inserting documents...")
        await rag.ainsert("\n\n".join(documents))
        
        # Query using different retrieval modes
        modes = ["naive", "local", "global", "hybrid"]
        query = "Giải thích mối quan hệ giữa AI, Máy học, và Học sâu. Trả lời bằng tiếng Việt."
        system_prompt = "Bạn là trợ lý AI chuyên trả lời bằng tiếng Việt. Luôn trả lời bằng tiếng Việt."
        for mode in modes:
            print(f"\n{'='*50}")
            print(f"Results using {mode} mode:")
            print('='*50)
            try:
                result = await rag.aquery(
                    query, 
                    param=QueryParam(mode=mode, enable_rerank=False),
                    system_prompt=system_prompt
                )
                print(result)
            except Exception as e:
                print(f"Error in {mode} mode: {e}")
    
        # === Lưu Knowledge Graph vào Neo4j sau khi insert và query ===
        print("\nĐang import KG vào Neo4j database 'lightrag'...")

        NEO4J_URI = "neo4j://localhost:7687"
        NEO4J_USERNAME = "neo4j"
        NEO4J_PASSWORD = "huy1552004"
        NEO4J_DATABASE = "lightrag"

        graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), name=NEO4J_DATABASE)
        graphml_path = "./my_rag_project/graph_chunk_entity_relation.graphml"

        if os.path.exists(graphml_path):
            tree = ET.parse(graphml_path)
            root = tree.getroot()
            ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}

            # Xóa dữ liệu cũ
            graph.run("MATCH (n) DETACH DELETE n")

            # Import nodes
            nodes = root.findall('.//graphml:node', ns)
            for node in nodes:
                node_id = node.get('id')
                node_data = {}
                for data in node.findall('graphml:data', ns):
                    key = data.get('key')
                    node_data[key] = data.text
                n = Node("Entity", name=node_id, **node_data)
                graph.create(n)

            # Import edges
            edges = root.findall('.//graphml:edge', ns)
            for edge in edges:
                source_id = edge.get('source')
                target_id = edge.get('target')
                edge_data = {}
                for data in edge.findall('graphml:data', ns):
                    key = data.get('key')
                    edge_data[key] = data.text
                source_node = graph.nodes.match("Entity", name=source_id).first()
                target_node = graph.nodes.match("Entity", name=target_id).first()
                if source_node and target_node:
                    rel = Relationship(source_node, "RELATED", target_node, **edge_data)
                    graph.create(rel)

            print("✅ Đã lưu KG vào Neo4j database 'lightrag'.")
            print("Mở Neo4j Browser, chọn DB 'lightrag' và chạy:")
            print("MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 25")
        else:
            print(f"Không tìm thấy file: {graphml_path}")

    # Chạy async
    asyncio.run(main())