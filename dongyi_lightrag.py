from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
import asyncio

# Khởi tạo LightRAG
rag = LightRAG(
    working_dir="./lightrag_dongyi",
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3.2:latest",
    embedding_func=ollama_embedding,
    embedding_model_name="nomic-embed-text:latest"
)

# Insert documents
async def build_kg():
    with open("./data/dongyi_data.txt", "r", encoding="utf-8") as f:
        await rag.ainsert(f.read())

# Query
async def query(question: str):
    result = await rag.aquery(
        question,
        param=QueryParam(mode="hybrid")  # hybrid = graph + vector
    )
    print(result)

# Run
asyncio.run(build_kg())
asyncio.run(query("Bài thuốc chữa sốt cao"))