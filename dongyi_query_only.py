# Hệ thống RAG Đông y - Neo4j + Ollama
# ------------------------------------------------
import os
import asyncio
import logging
import traceback
from neo4j import GraphDatabase
from typing import List, Dict
import json
import re

# Thêm import cho Ollama
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("⚠️  requests chưa cài đặt. Chạy: pip install requests")

# --- Cấu hình ---
print("--- Hệ thống RAG Đông y (Neo4j + Ollama) ---")

# Neo4j Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "huy1552004"
NEO4J_DATABASE = "dongyi"

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:latest"  # Sửa thành llama3.2:latest

print(f"✓ Neo4j Database: {NEO4J_DATABASE}")
print(f"✓ Ollama URL: {OLLAMA_BASE_URL}")
print(f"✓ Ollama Model: {OLLAMA_MODEL}")

# --- Query Preprocessor ---
class QueryPreprocessor:
    """Xử lý câu hỏi để trích xuất từ khóa"""
    
    # Danh sách stop words tiếng Việt
    STOP_WORDS = {
        'bài', 'thuốc', 'nào', 'trị', 'chữa', 'điều', 'trị', 'có', 'để',
        'là', 'gì', 'thế', 'như', 'thì', 'được', 'của', 'cho', 'và',
        'một', 'các', 'này', 'kia', 'đó', 'ấy', 'mà', 'với', 'hay',
        'hoặc', 'nhưng', 'tôi', 'muốn', 'cần', 'tìm', 'kiếm', 'xem',
        'biết', 'hỏi', 'giúp', 'em', 'anh', 'chị'
    }
    
    # Các từ liên quan đến bệnh
    DISEASE_KEYWORDS = {
        'sốt', 'ho', 'viêm', 'đau', 'cảm', 'nhiễm', 'lạnh', 'nóng',
        'khó', 'tiêu', 'táo', 'bón', 'tiêu', 'chảy', 'kiết', 'lỵ',
        'mệt', 'nhức', 'đầu', 'họng', 'phổi', 'gan', 'thận', 'tim'
    }
    
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """Trích xuất từ khóa quan trọng từ câu hỏi"""
        # Lowercase
        query = query.lower().strip()
        
        # Tách từ
        words = re.findall(r'\w+', query)
        
        # Lọc stop words
        keywords = [w for w in words if w not in QueryPreprocessor.STOP_WORDS and len(w) > 1]
        
        # Nếu không còn keyword nào, return query gốc
        if not keywords:
            return [query]
        
        # Ưu tiên các keyword về bệnh
        disease_keywords = [k for k in keywords if k in QueryPreprocessor.DISEASE_KEYWORDS]
        if disease_keywords:
            return disease_keywords
        
        return keywords
    
    @staticmethod
    def build_search_patterns(query: str) -> List[str]:
        """Tạo nhiều pattern search từ query"""
        keywords = QueryPreprocessor.extract_keywords(query)
        
        patterns = []
        
        # Pattern 1: Tất cả keywords ghép lại
        if len(keywords) > 1:
            patterns.append(' '.join(keywords))
        
        # Pattern 2: Từng keyword riêng lẻ
        patterns.extend(keywords)
        
        # Pattern 3: Query gốc
        patterns.append(query.lower().strip())
        
        # Loại bỏ duplicate
        return list(dict.fromkeys(patterns))


# --- Ollama Service ---
class OllamaService:
    """Service để gọi Ollama local LLM"""
    
    def __init__(self, base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Kiểm tra kết nối Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                print(f"✓ Kết nối Ollama thành công")
                print(f"  Models có sẵn: {', '.join(model_names)}")
                
                if self.model not in model_names:
                    print(f"⚠️  Model '{self.model}' chưa được pull")
                    print(f"   Model có sẵn gần nhất: {model_names[0] if model_names else 'không có'}")
                    # Tự động sử dụng model đầu tiên
                    if model_names:
                        self.model = model_names[0]
                        print(f"   ✓ Tự động chuyển sang model: {self.model}")
                    else:
                        raise ValueError(f"Không có model nào. Chạy: ollama pull llama3.2")
            else:
                raise ConnectionError("Không thể kết nối Ollama")
        except requests.exceptions.RequestException as e:
            print(f"✗ Không kết nối được Ollama tại {self.base_url}")
            print(f"  Lỗi: {e}")
            print("\n📌 HƯỚNG DẪN CÀI ĐẶT OLLAMA:")
            print("  1. Tải Ollama: https://ollama.ai/download")
            print("  2. Cài đặt và chạy Ollama")
            print("  3. Pull model: ollama pull llama3.2")
            print("  4. Kiểm tra: ollama list")
            raise
    
    def generate_answer(self, question: str, context: List[Dict]) -> str:
        """Sinh câu trả lời từ context sử dụng Ollama"""
        try:
            # Format context
            context_text = self._format_context(context)
            
            # Tạo prompt
            prompt = f"""Bạn là chuyên gia Y học Đông y Việt Nam. Dựa trên thông tin sau đây từ cơ sở tri thức, hãy trả lời câu hỏi của người dùng một cách chi tiết, chuyên nghiệp và dễ hiểu.

THÔNG TIN TỪ CƠ SỞ TRI THỨC:
{context_text}

CÂU HỎI: {question}

HƯỚNG DẪN TRẢ LỜI:
- Trả lời bằng tiếng Việt, chuyên nghiệp và dễ hiểu
- Nêu rõ tên bài thuốc, nguyên liệu, liều lượng
- Giải thích công hiệu và cách sử dụng
- Nếu có nhiều bài thuốc, so sánh và đưa ra khuyến nghị
- Luôn nhắc nhở "nên tham khảo ý kiến bác sĩ Đông y trước khi sử dụng"
- Nếu không có thông tin, hãy thành thật nói "Tôi không tìm thấy thông tin về..."

TRẢ LỜI:"""

            print("🤖 Đang gọi Ollama...")
            
            # Gọi Ollama API
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 1000
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120  # 2 minutes timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                # Debug info
                print(f"✓ Ollama response received ({len(answer)} chars)")
                return answer
            else:
                print(f"✗ Ollama API error: {response.status_code}")
                print(f"  Response: {response.text}")
                return self._fallback_answer(context)
                
        except requests.exceptions.Timeout:
            print("⚠️  Ollama timeout - model đang load hoặc quá chậm")
            return self._fallback_answer(context)
        except Exception as e:
            print(f"⚠️  Lỗi khi gọi Ollama: {e}")
            traceback.print_exc()
            return self._fallback_answer(context)
    
    def _format_context(self, context: List[Dict]) -> str:
        """Format context từ Neo4j thành text"""
        if not context:
            return "Không tìm thấy thông tin liên quan."
        
        formatted = []
        for i, item in enumerate(context, 1):
            text = f"\n--- BÀI THUỐC {i}: {item['ten_bai_thuoc']} ---\n"
            text += item['description']
            formatted.append(text)
        
        return "\n".join(formatted)
    
    def _fallback_answer(self, context: List[Dict]) -> str:
        """Câu trả lời dự phòng khi Ollama lỗi"""
        if not context:
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở tri thức."
        
        answer = "Dựa trên cơ sở tri thức, tôi tìm thấy các thông tin sau:\n\n"
        for i, item in enumerate(context, 1):
            answer += f"**{i}. {item['ten_bai_thuoc']}**\n"
            answer += f"{item['description']}\n\n"
        
        answer += "\n⚠️  *Lưu ý: Nên tham khảo ý kiến bác sĩ Đông y trước khi sử dụng.*"
        return answer


# --- Neo4j Query Helper (Cải tiến) ---
class DongyiQueryHelper:
    def __init__(self, uri, username, password, database="dongyi"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.preprocessor = QueryPreprocessor()
        
    def close(self):
        self.driver.close()
    
    def query_dongyi_kg(self, query_text: str, limit: int = 5) -> List[Dict]:
        """Truy vấn Knowledge Graph Đông y - Trả về structured data"""
        try:
            with self.driver.session(database=self.database) as session:
                # Debug
                count_result = session.run("MATCH (n) RETURN count(n) as total")
                total_entities = count_result.single()["total"]
                print(f"   📊 Database có {total_entities} nodes")
                
                if total_entities == 0:
                    print(f"   ⚠️  Database '{self.database}' trống!")
                    return []
                
                # Trích xuất keywords
                search_patterns = self.preprocessor.build_search_patterns(query_text)
                print(f"   🔍 Tìm kiếm với keywords: {search_patterns[:3]}")
                
                # Thử search với từng pattern
                for pattern in search_patterns:
                    print(f"      → Thử pattern: '{pattern}'")
                    
                    # Chiến lược truy vấn đa tầng
                    queries = [
                        self._query_by_disease(pattern, limit),
                        self._query_by_ingredient(pattern, limit),
                        self._query_by_effect(pattern, limit),
                        self._query_by_remedy_name(pattern, limit)
                    ]
                    
                    # Thử từng query cho đến khi có kết quả
                    for query_func in queries:
                        try:
                            results = query_func(session)
                            if results:
                                return results
                        except Exception as e:
                            continue
                
                print("   ✗ Không tìm thấy kết quả với tất cả patterns")
                return []
                
        except Exception as e:
            print(f"❌ Lỗi truy vấn Neo4j: {e}")
            traceback.print_exc()
            return []
    
    def _query_by_disease(self, query_text: str, limit: int):
        """Query 1: Tìm theo bệnh - Xử lý NaN"""
        def execute(session):
            query = """
            MATCH (r:`BÀI THUỐC`)-[:`CHỮA TRỊ`]->(b:`BỆNH`)
            WHERE b.tên_bệnh IS NOT NULL 
              AND toString(b.tên_bệnh) <> 'NaN'
              AND toLower(toString(b.tên_bệnh)) CONTAINS toLower($query_text)
            OPTIONAL MATCH (r)-[:`CHỨA NGUYÊN LIỆU`]->(n:`NGUYÊN LIỆU`)
            WHERE n.tên_nguyên_liệu IS NOT NULL AND toString(n.tên_nguyên_liệu) <> 'NaN'
            OPTIONAL MATCH (r)-[:`CÓ CÔNG HIỆU`]->(e:`CÔNG HIỆU`)
            WHERE e.tên_công_hiệu IS NOT NULL AND toString(e.tên_công_hiệu) <> 'NaN'
            RETURN DISTINCT
                r.tên_bài_thuốc AS ten_bai_thuoc,
                b.tên_bệnh AS ten_benh,
                r.liều_lượng_cách_dùng AS lieu_luong,
                r.chú_ý AS chu_y,
                r.đối_tượng_phù_hợp AS doi_tuong,
                collect(DISTINCT n.tên_nguyên_liệu)[..10] AS nguyen_lieu,
                collect(DISTINCT e.tên_công_hiệu)[..5] AS cong_hieu
            LIMIT $limit
            """
            results = session.run(query, query_text=query_text, limit=limit)
            return self._format_results(results, "bệnh")
        return execute
    
    def _query_by_ingredient(self, query_text: str, limit: int):
        """Query 2: Tìm theo nguyên liệu - Xử lý NaN"""
        def execute(session):
            query = """
            MATCH (r:`BÀI THUỐC`)-[:`CHỨA NGUYÊN LIỆU`]->(n:`NGUYÊN LIỆU`)
            WHERE n.tên_nguyên_liệu IS NOT NULL 
              AND toString(n.tên_nguyên_liệu) <> 'NaN'
              AND toLower(toString(n.tên_nguyên_liệu)) CONTAINS toLower($query_text)
            OPTIONAL MATCH (r)-[:`CHỮA TRỊ`]->(b:`BỆNH`)
            WHERE b.tên_bệnh IS NOT NULL AND toString(b.tên_bệnh) <> 'NaN'
            OPTIONAL MATCH (r)-[:`CÓ CÔNG HIỆU`]->(e:`CÔNG HIỆU`)
            WHERE e.tên_công_hiệu IS NOT NULL AND toString(e.tên_công_hiệu) <> 'NaN'
            RETURN DISTINCT
                r.tên_bài_thuốc AS ten_bai_thuoc,
                n.tên_nguyên_liệu AS nguyen_lieu_chinh,
                r.liều_lượng_cách_dùng AS lieu_luong,
                collect(DISTINCT b.tên_bệnh)[..5] AS benh,
                collect(DISTINCT e.tên_công_hiệu)[..5] AS cong_hieu
            LIMIT $limit
            """
            results = session.run(query, query_text=query_text, limit=limit)
            return self._format_results(results, "nguyên liệu")
        return execute
    
    def _query_by_effect(self, query_text: str, limit: int):
        """Query 3: Tìm theo công hiệu - Xử lý NaN"""
        def execute(session):
            query = """
            MATCH (r:`BÀI THUỐC`)-[:`CÓ CÔNG HIỆU`]->(e:`CÔNG HIỆU`)
            WHERE e.tên_công_hiệu IS NOT NULL 
              AND toString(e.tên_công_hiệu) <> 'NaN'
              AND toLower(toString(e.tên_công_hiệu)) CONTAINS toLower($query_text)
            OPTIONAL MATCH (r)-[:`CHỮA TRỊ`]->(b:`BỆNH`)
            WHERE b.tên_bệnh IS NOT NULL AND toString(b.tên_bệnh) <> 'NaN'
            OPTIONAL MATCH (r)-[:`CHỨA NGUYÊN LIỆU`]->(n:`NGUYÊN LIỆU`)
            WHERE n.tên_nguyên_liệu IS NOT NULL AND toString(n.tên_nguyên_liệu) <> 'NaN'
            RETURN DISTINCT
                r.tên_bài_thuốc AS ten_bai_thuoc,
                e.tên_công_hiệu AS cong_hieu_chinh,
                r.liều_lượng_cách_dùng AS lieu_luong,
                collect(DISTINCT b.tên_bệnh)[..5] AS benh,
                collect(DISTINCT n.tên_nguyên_liệu)[..10] AS nguyen_lieu
            LIMIT $limit
            """
            results = session.run(query, query_text=query_text, limit=limit)
            return self._format_results(results, "công hiệu")
        return execute
    
    def _query_by_remedy_name(self, query_text: str, limit: int):
        """Query 4: Tìm theo tên bài thuốc - Xử lý NaN"""
        def execute(session):
            query = """
            MATCH (r:`BÀI THUỐC`)
            WHERE r.tên_bài_thuốc IS NOT NULL 
              AND toString(r.tên_bài_thuốc) <> 'NaN'
              AND toLower(toString(r.tên_bài_thuốc)) CONTAINS toLower($query_text)
            OPTIONAL MATCH (r)-[:`CHỮA TRỊ`]->(b:`BỆNH`)
            WHERE b.tên_bệnh IS NOT NULL AND toString(b.tên_bệnh) <> 'NaN'
            OPTIONAL MATCH (r)-[:`CHỨA NGUYÊN LIỆU`]->(n:`NGUYÊN LIỆU`)
            WHERE n.tên_nguyên_liệu IS NOT NULL AND toString(n.tên_nguyên_liệu) <> 'NaN'
            OPTIONAL MATCH (r)-[:`CÓ CÔNG HIỆU`]->(e:`CÔNG HIỆU`)
            WHERE e.tên_công_hiệu IS NOT NULL AND toString(e.tên_công_hiệu) <> 'NaN'
            RETURN DISTINCT
                r.tên_bài_thuốc AS ten_bai_thuoc,
                r.liều_lượng_cách_dùng AS lieu_luong,
                r.chú_ý AS chu_y,
                collect(DISTINCT b.tên_bệnh)[..5] AS benh,
                collect(DISTINCT n.tên_nguyên_liệu)[..10] AS nguyen_lieu,
                collect(DISTINCT e.tên_công_hiệu)[..5] AS cong_hieu
            LIMIT $limit
            """
            results = session.run(query, query_text=query_text, limit=limit)
            return self._format_results(results, "tên bài thuốc")
        return execute
    
    def _format_results(self, results, query_type: str) -> List[Dict]:
        """Format kết quả từ Neo4j - Xử lý NaN"""
        entities = []
        for record in results:
            ten_bai = record.get('ten_bai_thuoc', 'N/A')
            
            # Skip nếu tên bài thuốc là NaN
            if not ten_bai or str(ten_bai) == 'NaN':
                continue
            
            # Build description
            description_parts = []
            
            # Bệnh
            if 'ten_benh' in record and record['ten_benh'] and str(record['ten_benh']) != 'NaN':
                description_parts.append(f"**Chữa bệnh:** {record['ten_benh']}")
            elif 'benh' in record:
                benh_list = [b for b in record.get('benh', []) if b and str(b) != 'NaN']
                if benh_list:
                    description_parts.append(f"**Chữa bệnh:** {', '.join(benh_list)}")
            
            # Nguyên liệu
            if 'nguyen_lieu_chinh' in record and record['nguyen_lieu_chinh'] and str(record['nguyen_lieu_chinh']) != 'NaN':
                description_parts.append(f"**Nguyên liệu chính:** {record['nguyen_lieu_chinh']}")
            
            nguyen_lieu = []
            if 'nguyen_lieu' in record:
                nguyen_lieu = [nl for nl in record.get('nguyen_lieu', []) if nl and str(nl) != 'NaN']
            if nguyen_lieu:
                description_parts.append(f"**Thành phần:** {', '.join(nguyen_lieu)}")
            
            # Công hiệu
            if 'cong_hieu_chinh' in record and record['cong_hieu_chinh'] and str(record['cong_hieu_chinh']) != 'NaN':
                description_parts.append(f"**Công hiệu chính:** {record['cong_hieu_chinh']}")
            
            cong_hieu = []
            if 'cong_hieu' in record:
                cong_hieu = [ch for ch in record.get('cong_hieu', []) if ch and str(ch) != 'NaN']
            if cong_hieu:
                description_parts.append(f"**Các công hiệu:** {', '.join(cong_hieu)}")
            
            # Liều lượng
            lieu_luong = record.get('lieu_luong', '')
            if lieu_luong and isinstance(lieu_luong, str) and str(lieu_luong) != 'NaN':
                description_parts.append(f"**Liều lượng & Cách dùng:** {lieu_luong[:500]}...")
            
            # Chú ý
            chu_y = record.get('chu_y', '')
            if chu_y and isinstance(chu_y, str) and str(chu_y) != 'NaN':
                description_parts.append(f"**Chú ý:** {chu_y[:300]}...")
            
            # Đối tượng
            doi_tuong = record.get('doi_tuong', '')
            if doi_tuong and isinstance(doi_tuong, str) and str(doi_tuong) != 'NaN':
                description_parts.append(f"**Đối tượng phù hợp:** {doi_tuong}")
            
            # Chỉ thêm nếu có ít nhất một thông tin
            if description_parts:
                entities.append({
                    'ten_bai_thuoc': ten_bai,
                    'description': '\n'.join(description_parts),
                    'query_type': query_type
                })
        
        if entities:
            print(f"   ✓ Tìm thấy {len(entities)} bài thuốc (theo {query_type})")
        
        return entities


# --- RAG System ---
async def interactive_rag_query():
    """Hệ thống RAG tương tác - Neo4j + Ollama"""
    print("\n" + "="*70)
    print("🏥 HỆ THỐNG RAG TRA CỨU ĐÔNG Y (OLLAMA)")
    print("="*70)
    print("Nhập 'exit' để thoát, 'help' để xem hướng dẫn")
    print("Nhập 'mode' để chuyển chế độ (rag/raw)")
    print("="*70 + "\n")
    
    # Khởi tạo Neo4j
    try:
        neo4j_helper = DongyiQueryHelper(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)
        with neo4j_helper.driver.session(database=neo4j_helper.database) as session:
            session.run("RETURN 1")
        print("✓ Kết nối Neo4j thành công")
    except Exception as e:
        print(f"✗ Không kết nối được Neo4j: {e}")
        return
    
    # Khởi tạo Ollama
    ollama_service = None
    try:
        ollama_service = OllamaService()
        print("✓ Kết nối Ollama thành công\n")
    except Exception as e:
        print(f"⚠️  Không kết nối được Ollama: {e}")
        print("   Sẽ chạy ở chế độ 'raw' (chỉ hiển thị dữ liệu thô)\n")
    
    mode = "rag" if ollama_service else "raw"
    
    try:
        while True:
            try:
                print("─" * 70)
                print(f"[Chế độ: {mode.upper()}] [Model: {OLLAMA_MODEL}]")
                user_query = input("💬 Câu hỏi: ").strip()
                
                if user_query.lower() == 'exit':
                    print("\n👋 Tạm biệt!")
                    break
                elif user_query.lower() == 'help':
                    print_help()
                    continue
                elif user_query.lower() == 'mode':
                    if ollama_service:
                        mode = "raw" if mode == "rag" else "rag"
                        print(f"✓ Đã chuyển sang chế độ: {mode.upper()}")
                    else:
                        print("⚠️  Ollama chưa kết nối, không thể dùng chế độ RAG")
                    continue
                elif not user_query:
                    print("⚠️  Vui lòng nhập câu hỏi!")
                    continue
                
                print(f"\n🔍 Đang tìm kiếm...\n")
                
                # Bước 1: Truy vấn Neo4j
                context = neo4j_helper.query_dongyi_kg(user_query, limit=5)
                
                if not context:
                    print("❌ Không tìm thấy thông tin liên quan")
                    print("💡 Thử từ khóa: 'sốt', 'ho', 'đau đầu', 'lá tre', 'thạch cao'\n")
                    continue
                
                # Bước 2: Sinh câu trả lời
                if mode == "rag" and ollama_service:
                    print("=" * 70)
                    answer = ollama_service.generate_answer(user_query, context)
                    print(answer)
                    print("=" * 70)
                else:
                    # Chế độ RAW - hiển thị dữ liệu thô
                    print(f"📋 KẾT QUẢ TÌM KIẾM ({len(context)} bài thuốc):\n")
                    for i, entity in enumerate(context, 1):
                        print(f"{'─'*70}")
                        print(f"🔹 BÀI THUỐC {i}: {entity['ten_bai_thuoc']}")
                        print(f"{'─'*70}")
                        print(entity['description'])
                        print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"❌ Lỗi: {e}")
                traceback.print_exc()
        
    finally:
        neo4j_helper.close()
        print("\n✓ Đã đóng kết nối Neo4j")


def print_help():
    """In hướng dẫn"""
    print("\n" + "="*70)
    print("📖 HƯỚNG DẪN SỬ DỤNG")
    print("="*70)
    print("• Nhập câu hỏi về Đông y để tìm kiếm bài thuốc")
    print("• 'help' - Xem hướng dẫn")
    print("• 'mode' - Chuyển đổi giữa chế độ RAG (có LLM) và RAW (dữ liệu thô)")
    print("• 'exit' - Thoát chương trình")
    print("\n🎯 CHẾ ĐỘ:")
    print("   RAG  - Sử dụng Ollama để sinh câu trả lời tự nhiên")
    print("   RAW  - Hiển thị dữ liệu thô từ Neo4j")
    print("\n💡 VÍ DỤ CÂU HỎI:")
    print("   - Bài thuốc chữa sốt")
    print("   - Thuốc nào có lá tre")
    print("   - Công hiệu thanh nhiệt")
    print("   - Chữa ho cho trẻ em")
    print("   - Nguyên liệu thạch cao dùng để làm gì")
    print("   - Bài thuốc nào trị sốt cao")
    print("\n📌 CÀI ĐẶT OLLAMA:")
    print("   1. Tải: https://ollama.ai/download")
    print("   2. Cài đặt và chạy Ollama")
    print("   3. Pull model: ollama pull llama3.2")
    print("   4. Kiểm tra: ollama list")
    print("="*70 + "\n")


async def main():
    """Hàm chính"""
    try:
        await interactive_rag_query()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.ERROR)
        asyncio.run(main())
        print("\n✓ Chương trình hoàn tất!\n")
    except KeyboardInterrupt:
        print("\n⚠️  Dừng bởi người dùng\n")
    except Exception as e:
        print(f"\n✗ Lỗi: {e}\n")