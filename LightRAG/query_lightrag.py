import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

WORKING_DIR = "./lightrag_dongyi_neo4j"

async def sentence_transformer_embedding(texts: list[str]):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, convert_to_numpy=True)

# Nếu muốn dùng Ollama để sinh câu trả lời, thêm hàm này:
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

async def main():
    print("🔍 Truy vấn LightRAG DB Đông y")
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
    # Bỏ dòng này: await rag.load_storages()

    while True:
        query = input("\nNhập câu hỏi Đông y (hoặc 'exit' để thoát): ").strip()
        if query.lower() == "exit":
            break
        try:
            result = await rag.aquery(
                query,
                param=QueryParam(
                    mode="naive",
                    only_need_context=False, 
                    top_k=3
                )
            )
            print("\nKết quả truy vấn:")
            print(result)
        except Exception as e:
            print(f"Lỗi truy vấn: {e}")

    await rag.close_storages()

if __name__ == "__main__":
    asyncio.run(main())