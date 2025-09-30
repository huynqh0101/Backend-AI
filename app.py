from flask import Flask, request, jsonify
from flask_cors import CORS
import time

# Khởi tạo Flask app
app = Flask(__name__)
# Cho phép Cross-Origin Resource Sharing (CORS) để NestJS có thể gọi qua
CORS(app)

def get_lightrag_response(question: str):
    """
    TODO: ĐÂY LÀ NƠI BẠN SẼ TÍCH HỢP LOGIC LightRAG THỰC TẾ
    - Nhận `question` làm đầu vào.
    - Thực hiện các bước: vector hóa, tìm kiếm ngữ cảnh, gọi LLM...
    - Trả về một dictionary có cấu trúc giống như dưới đây.
    """
    print(f"Received question: {question}")
    print("Simulating RAG processing...")
    time.sleep(2) # Giả lập thời gian xử lý của AI

    # Cấu trúc trả về phải nhất quán để NestJS có thể xử lý
    return {
        "answer": f"Đây là câu trả lời được tạo bởi LightRAG cho câu hỏi: '{question}'",
        "sources": [
            { "document": "TaiLieuSo1.pdf", "page": 4, "score": 0.91 },
            { "document": "HuongDanSuDung.docx", "page": 1, "score": 0.85 }
        ]
    }

@app.route("/generate", methods=["POST"])
def generate_response():
    # Lấy dữ liệu JSON từ request của NestJS
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    user_question = data['question']

    # Gọi hàm xử lý RAG của bạn
    rag_result = get_lightrag_response(user_question)

    # Trả về kết quả dưới dạng JSON
    return jsonify(rag_result)

if __name__ == "__main__":
    # Chạy server ở port 5001 để không bị trùng với NestJS
    app.run(host="0.0.0.0", port=5001, debug=True)