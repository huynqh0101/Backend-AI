# Hệ thống RAG Đông y - Neo4j + Ollama (CẬP NHẬT CHO KG MỚI)
# ------------------------------------------------
import os
import asyncio
import logging
import traceback
from neo4j import GraphDatabase
from typing import List, Dict
import json
import re
import unicodedata
import requests

# --- Cấu hình ---
print("--- Hệ thống RAG Đông y (Neo4j + Ollama) - KG V2 ---")

# Neo4j Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "huy1552004"
NEO4J_DATABASE = "dongyi"

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:latest"

print(f"✓ Neo4j Database: {NEO4J_DATABASE}")
print(f"✓ Ollama URL: {OLLAMA_BASE_URL}")
print(f"✓ Ollama Model: {OLLAMA_MODEL}")

# --- Text Normalizer ---
class TextNormalizer:
    """Chuẩn hóa text để tìm kiếm tốt hơn"""
    
    @staticmethod
    def remove_accents(text: str) -> str:
        """Bỏ dấu tiếng Việt"""
        if not isinstance(text, str):
            return ""
        
        # Normalize Unicode (NFD = tách ký tự và dấu)
        nfd = unicodedata.normalize('NFD', text)
        
        # Loại bỏ các dấu (Mn = Mark, Nonspacing)
        without_accents = ''.join(
            char for char in nfd 
            if unicodedata.category(char) != 'Mn'
        )
        
        # Xử lý Đ/đ đặc biệt
        without_accents = without_accents.replace('Đ', 'D').replace('đ', 'd')
        
        return without_accents
    
    @staticmethod
    def normalize(text: str, keep_case: bool = False) -> str:
        """Chuẩn hóa text toàn diện"""
        if not isinstance(text, str):
            return ""
        
        # 1. Loại bỏ ký tự đặc biệt (giữ chữ, số, khoảng trắng)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 2. Chuẩn hóa khoảng trắng (nhiều space → 1 space)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 3. Lowercase (trừ khi keep_case=True)
        if not keep_case:
            text = text.lower()
        
        return text
    
    @staticmethod
    def normalize_for_search(text: str) -> str:
        """Chuẩn hóa cho tìm kiếm: bỏ dấu + lowercase + trim"""
        text = TextNormalizer.remove_accents(text)
        text = TextNormalizer.normalize(text, keep_case=False)
        return text
    
    @staticmethod
    def create_search_variants(text: str) -> list:
        """Tạo các biến thể để tìm kiếm"""
        variants = set()
        
        # Variant 1: Gốc
        variants.add(text.strip())
        
        # Variant 2: Lowercase
        variants.add(text.lower().strip())
        
        # Variant 3: Bỏ dấu
        variants.add(TextNormalizer.remove_accents(text).lower().strip())
        
        # Variant 4: Normalize hoàn toàn
        variants.add(TextNormalizer.normalize_for_search(text))
        
        # Variant 5: Bỏ "cây" ở đầu
        if text.lower().startswith('cây '):
            variants.add(text[4:].strip())
            variants.add(TextNormalizer.normalize_for_search(text[4:]))
        
        return list(variants)


# --- Query Preprocessor (SỬA ĐỔI) ---
class QueryPreprocessor:
    """Xử lý câu hỏi để trích xuất từ khóa"""
    
    STOP_WORDS = {
        'bài', 'thuốc', 'nào', 'trị', 'chữa', 'điều', 'có', 'để',
        'là', 'gì', 'thế', 'như', 'thì', 'được', 'của', 'cho', 'và',
        'một', 'các', 'này', 'kia', 'đó', 'ấy', 'mà', 'với', 'hay',
        'hoặc', 'nhưng', 'tôi', 'muốn', 'cần', 'tìm', 'kiếm', 'xem',
        'biết', 'hỏi', 'giúp', 'em', 'anh', 'chị'
    }
    
    DISEASE_KEYWORDS = {
        'sốt', 'ho', 'viêm', 'đau', 'cảm', 'nhiễm', 'lạnh', 'nóng',
        'khó', 'tiêu', 'táo', 'bón', 'chảy', 'kiết', 'lỵ',
        'mệt', 'nhức', 'đầu', 'họng', 'phổi', 'gan', 'thận', 'tim',
        'khát', 'phiền', 'buồn'
    }
    
    HERB_KEYWORDS = {
        'cây', 'thảo', 'dược', 'liệu', 'họ', 'thực', 'vật', 'lá', 'rễ', 
        'thân', 'hoa', 'quả', 'củ', 'vỏ'
    }
    
    @staticmethod
    def detect_query_type(query: str) -> str:
        """Phát hiện loại câu hỏi - CẢI TIẾN"""
        query_normalized = TextNormalizer.normalize_for_search(query)
        
        # Check herb keywords
        herb_count = sum(1 for kw in QueryPreprocessor.HERB_KEYWORDS 
                        if kw in query_normalized)
        disease_count = sum(1 for kw in QueryPreprocessor.DISEASE_KEYWORDS 
                           if kw in query_normalized)
        
        if herb_count > disease_count:
            return "herb"
        elif disease_count > 0:
            return "disease"
        else:
            return "general"
    
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """Trích xuất keywords - CẢI TIẾN"""
        query_normalized = TextNormalizer.normalize_for_search(query)
        words = re.findall(r'\w+', query_normalized)
        
        # Loại bỏ stop words (đã normalize)
        stop_words_normalized = {TextNormalizer.normalize_for_search(w) 
                                 for w in QueryPreprocessor.STOP_WORDS}
        keywords = [w for w in words 
                   if w not in stop_words_normalized and len(w) > 1]
        
        if not keywords:
            return [query_normalized]
        
        # Ưu tiên disease keywords
        disease_keywords_normalized = {TextNormalizer.normalize_for_search(w) 
                                       for w in QueryPreprocessor.DISEASE_KEYWORDS}
        disease_found = [k for k in keywords if k in disease_keywords_normalized]
        if disease_found:
            return disease_found
        
        return keywords
    
    @staticmethod
    def build_search_patterns(query: str) -> List[str]:
        """Tạo search patterns - CẢI TIẾN"""
        patterns = set()
        
        # Pattern 1: Nguyên gốc (trim)
        patterns.add(query.strip())
        
        # Pattern 2: Lowercase
        patterns.add(query.lower().strip())
        
        # Pattern 3: Normalize (bỏ dấu)
        patterns.add(TextNormalizer.normalize_for_search(query))
        
        # Pattern 4: Từ keywords
        keywords = QueryPreprocessor.extract_keywords(query)
        if len(keywords) > 1:
            patterns.add(' '.join(keywords))
        patterns.update(keywords)
        
        # Pattern 5: Variants (bỏ "cây", "thuốc"...)
        variants = TextNormalizer.create_search_variants(query)
        patterns.update(variants)
        
        # Loại bỏ empty và trùng lặp, giữ thứ tự
        result = []
        for p in patterns:
            p_clean = p.strip()
            if p_clean and p_clean not in result:
                result.append(p_clean)
        
        return result


# --- Ollama Service ---
class OllamaService:
    """Service để gọi Ollama local LLM"""
    
    def __init__(self, base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"
        self._test_connection()
    
    def _test_connection(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                print(f"✓ Kết nối Ollama thành công")
                print(f"  Models có sẵn: {', '.join(model_names)}")
                
                if self.model not in model_names:
                    print(f"⚠️  Model '{self.model}' chưa được pull")
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
            raise
    
    def generate_answer(self, question: str, context: List[Dict]) -> str:
        try:
            context_text = self._format_context(context)
            
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
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 1000
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                print(f"✓ Ollama response received ({len(answer)} chars)")
                return answer
            else:
                return self._fallback_answer(context)
                
        except Exception as e:
            print(f"⚠️  Lỗi khi gọi Ollama: {e}")
            return self._fallback_answer(context)
    
    def _format_context(self, context: List[Dict]) -> str:
        if not context:
            return "Không tìm thấy thông tin liên quan."
        
        formatted = []
        for i, item in enumerate(context, 1):
            text = f"\n--- BÀI THUỐC {i}: {item['ten_bai_thuoc']} ---\n"
            text += item['description']
            formatted.append(text)
        
        return "\n".join(formatted)
    
    def _fallback_answer(self, context: List[Dict]) -> str:
        if not context:
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở tri thức."
        
        answer = "Dựa trên cơ sở tri thức, tôi tìm thấy các thông tin sau:\n\n"
        for i, item in enumerate(context, 1):
            answer += f"**{i}. {item['ten_bai_thuoc']}**\n"
            answer += f"{item['description']}\n\n"
        
        answer += "\n⚠️  *Lưu ý: Nên tham khảo ý kiến bác sĩ Đông y trước khi sử dụng.*"
        return answer


# --- Neo4j Query Helper (SỬA QUERY) ---
class DongyiQueryHelper:
    def __init__(self, uri, username, password, database="dongyi"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.preprocessor = QueryPreprocessor()
        self.normalizer = TextNormalizer()  # ← THÊM
        
    def close(self):
        self.driver.close()
    
    def _query_by_disease(self, query_text: str, limit: int):
        """Query 1: Tìm theo BỆNH - IMPROVED"""
        def execute(session):
            variants = TextNormalizer.create_search_variants(query_text)
            query = """
            MATCH (r:BÀI_THUỐC)-[:ĐIỀU_TRỊ]->(b:BỆNH)
            WHERE toLower(b.tên_bệnh) CONTAINS toLower($query_text)
               OR ANY(variant IN $variants WHERE toLower(b.tên_bệnh) CONTAINS toLower(variant))
            OPTIONAL MATCH (r)-[rel:CHỨA_NGUYÊN_LIỆU]->(n:NGUYÊN_LIỆU)
            OPTIONAL MATCH (r)-[:CÓ_CÔNG_HIỆU]->(e:CÔNG_HIỆU)
            OPTIONAL MATCH (r)-[:TRỊ_TRIỆU_CHỨNG]->(s:TRIỆU_CHỨNG)
            RETURN DISTINCT
                r.tên_bài_thuốc AS ten_bai_thuoc,
                b.tên_bệnh AS ten_benh,
                r.liều_lượng_cách_dùng AS lieu_luong,
                r.chú_ý AS chu_y,
                r.đối_tượng_phù_hợp AS doi_tuong,
                collect(DISTINCT n.tên_nguyên_liệu) AS nguyen_lieu,
                collect(DISTINCT e.tên_công_hiệu) AS cong_hieu,
                collect(DISTINCT s.mô_tả) AS trieu_chung
            LIMIT $limit
            """
            results = session.run(query, query_text=query_text, variants=variants, limit=limit)
            return self._format_results(results, "bệnh")
        return execute
    
    def _query_by_symptom(self, query_text: str, limit: int):
        """Query 2: Tìm theo TRIỆU_CHỨNG"""
        def execute(session):
            query = """
            MATCH (r:BÀI_THUỐC)-[:TRỊ_TRIỆU_CHỨNG]->(s:TRIỆU_CHỨNG)
            WHERE toLower(s.mô_tả) CONTAINS toLower($query_text)
            OPTIONAL MATCH (r)-[:ĐIỀU_TRỊ]->(b:BỆNH)
            OPTIONAL MATCH (r)-[rel:CHỨA_NGUYÊN_LIỆU]->(n:NGUYÊN_LIỆU)
            OPTIONAL MATCH (r)-[:CÓ_CÔNG_HIỆU]->(e:CÔNG_HIỆU)
            RETURN DISTINCT
                r.tên_bài_thuốc AS ten_bai_thuoc,
                s.mô_tả AS trieu_chung_chinh,
                r.liều_lượng_cách_dùng AS lieu_luong,
                collect(DISTINCT b.tên_bệnh) AS benh,
                collect(DISTINCT n.tên_nguyên_liệu) AS nguyen_lieu,
                collect(DISTINCT e.tên_công_hiệu) AS cong_hieu
            LIMIT $limit
            """
            results = session.run(query, query_text=query_text, limit=limit)
            return self._format_results(results, "triệu chứng")
        return execute
    
    def _query_by_ingredient(self, query_text: str, limit: int):
        """Query 3: Tìm theo NGUYÊN_LIỆU - FIXED"""
        def execute(session):
            query = """
            MATCH (r:BÀI_THUỐC)-[rel:CHỨA_NGUYÊN_LIỆU]->(n:NGUYÊN_LIỆU)
            WHERE toLower(n.tên_nguyên_liệu) CONTAINS toLower($query_text)
            OPTIONAL MATCH (r)-[:ĐIỀU_TRỊ]->(b:BỆNH)
            OPTIONAL MATCH (r)-[:CÓ_CÔNG_HIỆU]->(e:CÔNG_HIỆU)
            OPTIONAL MATCH (n)-[:LÀ_DƯỢC_LIỆU_TỪ]->(c:CÂY_THUỐC)
            RETURN DISTINCT
                r.tên_bài_thuốc AS ten_bai_thuoc,
                n.tên_nguyên_liệu AS nguyen_lieu_chinh,
                r.liều_lượng_cách_dùng AS lieu_luong,  // ← SỬA: Lấy từ node BÀI_THUỐC
                c.tên_chính AS cay_thuoc,
                c.tính_vị_tác_dụng AS tinh_vi,
                collect(DISTINCT b.tên_bệnh) AS benh,
                collect(DISTINCT e.tên_công_hiệu) AS cong_hieu
            LIMIT $limit
            """
            results = session.run(query, query_text=query_text, limit=limit)
            return self._format_results(results, "nguyên liệu")
        return execute
    
    def _query_by_effect(self, query_text: str, limit: int):
        """Query 4: Tìm theo CÔNG_HIỆU"""
        def execute(session):
            query = """
            MATCH (r:BÀI_THUỐC)-[:CÓ_CÔNG_HIỆU]->(e:CÔNG_HIỆU)
            WHERE toLower(e.tên_công_hiệu) CONTAINS toLower($query_text)
            OPTIONAL MATCH (r)-[:ĐIỀU_TRỊ]->(b:BỆNH)
            OPTIONAL MATCH (r)-[rel:CHỨA_NGUYÊN_LIỆU]->(n:NGUYÊN_LIỆU)
            RETURN DISTINCT
                r.tên_bài_thuốc AS ten_bai_thuoc,
                e.tên_công_hiệu AS cong_hieu_chinh,
                r.liều_lượng_cách_dùng AS lieu_luong,
                collect(DISTINCT b.tên_bệnh) AS benh,
                collect(DISTINCT n.tên_nguyên_liệu) AS nguyen_lieu
            LIMIT $limit
            """
            results = session.run(query, query_text=query_text, limit=limit)
            return self._format_results(results, "công hiệu")
        return execute
    
    def _query_by_remedy_name(self, query_text: str, limit: int):
        """Query 5: Tìm theo tên BÀI_THUỐC"""
        def execute(session):
            query = """
            MATCH (r:BÀI_THUỐC)
            WHERE toLower(r.tên_bài_thuốc) CONTAINS toLower($query_text)
            OPTIONAL MATCH (r)-[:ĐIỀU_TRỊ]->(b:BỆNH)
            OPTIONAL MATCH (r)-[rel:CHỨA_NGUYÊN_LIỆU]->(n:NGUYÊN_LIỆU)
            OPTIONAL MATCH (r)-[:CÓ_CÔNG_HIỆU]->(e:CÔNG_HIỆU)
            RETURN DISTINCT
                r.tên_bài_thuốc AS ten_bai_thuoc,
                r.liều_lượng_cách_dùng AS lieu_luong,
                r.chú_ý AS chu_y,
                collect(DISTINCT b.tên_bệnh) AS benh,
                collect(DISTINCT n.tên_nguyên_liệu) AS nguyen_lieu,
                collect(DISTINCT e.tên_công_hiệu) AS cong_hieu
            LIMIT $limit
            """
            results = session.run(query, query_text=query_text, limit=limit)
            return self._format_results(results, "tên bài thuốc")
        return execute
    
    def _query_by_herb(self, query_text: str, limit: int):
        """Query 6: Tìm theo CÂY_THUỐC - IMPROVED"""
        def execute(session):
            # Tạo variants để tìm kiếm
            search_variants = TextNormalizer.create_search_variants(query_text)
            
            # Tìm với nhiều điều kiện
            query = """
            MATCH (c:CÂY_THUỐC)
            WHERE toLower(c.tên_chính) CONTAINS toLower($query_text)
               OR toLower(c.tên_khoa_học) CONTAINS toLower($query_text)
               OR toLower(c.họ) CONTAINS toLower($query_text)
               OR ANY(variant IN $variants WHERE toLower(c.tên_chính) CONTAINS toLower(variant))
               OR toLower(c.tên_khác) CONTAINS toLower($query_text)
            
            OPTIONAL MATCH (c)<-[:LÀ_DƯỢC_LIỆU_TỪ]-(n:NGUYÊN_LIỆU)<-[:CHỨA_NGUYÊN_LIỆU]-(r:BÀI_THUỐC)
            OPTIONAL MATCH (c)-[:CÓ_TÊN_GỌI_KHÁC]->(tk:TÊN_KHÁC)
            OPTIONAL MATCH (c)-[:THUỘC_HỌ]->(h:HỌ_THỰC_VẬT)
            OPTIONAL MATCH (c)-[:SỬ_DỤNG_BỘ_PHẬN]->(bp:BỘ_PHẬN_DÙNG)
            OPTIONAL MATCH (c)-[:CHỨA_THÀNH_PHẦN]->(tp:THÀNH_PHẦN_HÓA_HỌC)
            
            RETURN DISTINCT
                c.tên_chính AS ten_cay_thuoc,
                c.tên_khoa_học AS ten_khoa_hoc,
                c.tên_khác AS ten_khac_str,
                c.họ AS ho,
                c.mô_tả AS mo_ta,
                c.nơi_sống_thu_hái AS noi_song,
                c.thành_phần_hóa_học AS thanh_phan_hoa_hoc,
                c.tính_vị_tác_dụng AS tinh_vi,
                c.công_dụng_chỉ_định AS cong_dung,
                c.liều_dùng AS lieu_dung,
                c.đơn_thuốc AS don_thuoc,
                collect(DISTINCT tk.tên) AS ten_khac,
                collect(DISTINCT h.tên_họ) AS ho_thuc_vat,
                collect(DISTINCT bp.tên_bộ_phận) AS cac_bo_phan,
                collect(DISTINCT tp.tên) AS cac_thanh_phan,
                collect(DISTINCT r.tên_bài_thuốc)[..5] AS bai_thuoc_su_dung
            LIMIT $limit
            """
            results = session.run(query, 
                                query_text=query_text, 
                                variants=search_variants,
                                limit=limit)
            return self._format_herb_results(results, "cây thuốc")
        return execute
    
    def _format_results(self, results, query_type: str) -> List[Dict]:
        """Format kết quả từ Neo4j - FIXED"""
        entities = []
        for record in results:
            ten_bai = record.get('ten_bai_thuoc', 'N/A')
            
            if not ten_bai or str(ten_bai) == 'None':
                continue
            
            description_parts = []
            
            # Bệnh
            if 'ten_benh' in record and record['ten_benh']:
                description_parts.append(f"**Chữa bệnh:** {record['ten_benh']}")
            elif 'benh' in record:
                benh_list = [b for b in record.get('benh', []) if b and str(b) != 'None']
                if benh_list:
                    description_parts.append(f"**Chữa bệnh:** {', '.join(benh_list)}")
            
            # Triệu chứng
            if 'trieu_chung_chinh' in record and record['trieu_chung_chinh']:
                description_parts.append(f"**Triệu chứng:** {record['trieu_chung_chinh']}")
            elif 'trieu_chung' in record:
                tc_list = [tc for tc in record.get('trieu_chung', []) if tc and str(tc) != 'None']
                if tc_list:
                    description_parts.append(f"**Triệu chứng:** {', '.join(tc_list)}")
            
            # Nguyên liệu
            if 'nguyen_lieu_chinh' in record and record['nguyen_lieu_chinh']:
                description_parts.append(f"**Nguyên liệu chính:** {record['nguyen_lieu_chinh']}")
                
                # Thông tin cây thuốc
                if record.get('cay_thuoc'):
                    description_parts.append(f"  - Nguồn gốc: {record['cay_thuoc']}")
                if record.get('tinh_vi'):
                    description_parts.append(f"  - Tính vị: {record['tinh_vi'][:200]}...")
            
            nguyen_lieu = [nl for nl in record.get('nguyen_lieu', []) if nl and str(nl) != 'None']
            if nguyen_lieu:
                description_parts.append(f"**Thành phần:** {', '.join(nguyen_lieu[:10])}")
            
            # Công hiệu
            if 'cong_hieu_chinh' in record and record['cong_hieu_chinh']:
                description_parts.append(f"**Công hiệu chính:** {record['cong_hieu_chinh']}")
            
            cong_hieu = [ch for ch in record.get('cong_hieu', []) if ch and str(ch) != 'None']
            if cong_hieu:
                description_parts.append(f"**Các công hiệu:** {', '.join(cong_hieu)}")
            
            # Liều lượng - SỬA ĐÂY
            lieu_luong = record.get('lieu_luong', '')
            if lieu_luong and str(lieu_luong) != 'None':
                # Rút ngắn nếu quá dài
                if len(str(lieu_luong)) > 500:
                    description_parts.append(f"**Liều lượng & Cách dùng:** {str(lieu_luong)[:500]}...")
                else:
                    description_parts.append(f"**Liều lượng & Cách dùng:** {lieu_luong}")
            
            # Chú ý
            chu_y = record.get('chu_y', '')
            if chu_y and str(chu_y) != 'None':
                description_parts.append(f"**Chú ý:** {str(chu_y)[:300]}...")
            
            # Đối tượng
            doi_tuong = record.get('doi_tuong', '')
            if doi_tuong and str(doi_tuong) != 'None':
                description_parts.append(f"**Đối tượng phù hợp:** {doi_tuong}")
            
            if description_parts:
                entities.append({
                    'ten_bai_thuoc': ten_bai,
                    'description': '\n'.join(description_parts),
                    'query_type': query_type
                })
        
        if entities:
            print(f"   ✓ Tìm thấy {len(entities)} bài thuốc (theo {query_type})")
        
        return entities

    def _format_herb_results(self, results, query_type: str) -> List[Dict]:
        """Format kết quả từ Neo4j cho CÂY_THUỐC"""
        entities = []
        for record in results:
            ten_cay = record.get('ten_cay_thuoc', 'N/A')
            
            if not ten_cay or str(ten_cay) == 'None':
                continue
            
            description_parts = []
            
            # Tên khoa học
            if record.get('ten_khoa_hoc'):
                description_parts.append(f"**Tên khoa học:** _{record['ten_khoa_hoc']}_")
            
            # Họ thực vật
            if record.get('ho'):
                description_parts.append(f"**Họ:** {record['ho']}")
            
            # Tên khác
            ten_khac = [tk for tk in record.get('ten_khac', []) if tk and str(tk) != 'None']
            if ten_khac:
                description_parts.append(f"**Tên gọi khác:** {', '.join(ten_khac)}")
            
            # Mô tả
            mo_ta = record.get('mo_ta', '')
            if mo_ta and str(mo_ta) != 'None':
                description_parts.append(f"**Mô tả:** {str(mo_ta)[:300]}...")
            
            # Bộ phận dùng
            if record.get('bo_phan_dung'):
                description_parts.append(f"**Bộ phận dùng:** {record['bo_phan_dung']}")
            
            # Nơi sống
            if record.get('noi_song'):
                description_parts.append(f"**Nơi sống và thu hái:** {str(record['noi_song'])[:200]}...")
            
            # Thành phần hóa học
            if record.get('thanh_phan_hoa_hoc'):
                description_parts.append(f"**Thành phần hóa học:** {str(record['thanh_phan_hoa_hoc'])[:200]}...")
            
            # Tính vị tác dụng
            tinh_vi = record.get('tinh_vi', '')
            if tinh_vi and str(tinh_vi) != 'None':
                description_parts.append(f"**Tính vị, tác dụng:** {str(tinh_vi)[:300]}...")
            
            # Liều dùng
            if record.get('lieu_dung'):
                description_parts.append(f"**Liều dùng:** {record['lieu_dung']}")
            
            # Bài thuốc sử dụng
            bai_thuoc = [bt for bt in record.get('bai_thuoc_su_dung', []) if bt and str(bt) != 'None']
            if bai_thuoc:
                description_parts.append(f"**Các bài thuốc sử dụng:** {', '.join(bai_thuoc)}")
            
            if description_parts:
                entities.append({
                    'ten_bai_thuoc': ten_cay,  # Giữ key này để tương thích với OllamaService
                    'description': '\n'.join(description_parts),
                    'query_type': query_type
                })
        
        if entities:
            print(f"   ✓ Tìm thấy {len(entities)} cây thuốc")
        
        return entities

    def query_dongyi_kg(self, query_text: str, limit: int = 5) -> List[Dict]:
        """Truy vấn Knowledge Graph V2 - Phát hiện thông minh"""
        try:
            with self.driver.session(database=self.database) as session:
                # Debug
                count_result = session.run("MATCH (n) RETURN count(n) as total")
                total_entities = count_result.single()["total"]
                print(f"   📊 Database có {total_entities} nodes")
                
                if total_entities == 0:
                    print(f"   ⚠️  Database '{self.database}' trống!")
                    return []
                
                # Phát hiện loại query
                query_type = self.preprocessor.detect_query_type(query_text)
                print(f"   🎯 Loại câu hỏi: {query_type.upper()}")
                
                # Trích xuất keywords
                search_patterns = self.preprocessor.build_search_patterns(query_text)
                print(f"   🔍 Tìm kiếm với keywords: {search_patterns[:3]}")
                
                for pattern in search_patterns:
                    print(f"      → Thử pattern: '{pattern}'")
                    
                    # Chọn thứ tự query dựa trên loại câu hỏi
                    if query_type == "herb":
                        queries = [
                            self._query_by_herb(pattern, limit),
                            self._query_by_ingredient(pattern, limit),
                            self._query_by_remedy_name(pattern, limit)
                        ]
                    else:
                        queries = [
                            self._query_by_disease(pattern, limit),
                            self._query_by_symptom(pattern, limit),
                            self._query_by_ingredient(pattern, limit),
                            self._query_by_effect(pattern, limit),
                            self._query_by_remedy_name(pattern, limit),
                            self._query_by_herb(pattern, limit)
                        ]
                
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

# --- RAG System ---
async def interactive_rag_query():
    """Hệ thống RAG tương tác - Neo4j + Ollama"""
    print("\n" + "="*70)
    print("🏥 HỆ THỐNG RAG TRA CỨU ĐÔNG Y V2 (OLLAMA)")
    print("="*70)
    print("Nhập 'exit' để thoát, 'help' để xem hướng dẫn")
    print("Nhập 'mode' để chuyển chế độ (rag/raw)")
    print("="*70 + "\n")
    
    try:
        neo4j_helper = DongyiQueryHelper(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)
        with neo4j_helper.driver.session(database=neo4j_helper.database) as session:
            session.run("RETURN 1")
        print("✓ Kết nối Neo4j thành công")
    except Exception as e:
        print(f"✗ Không kết nối được Neo4j: {e}")
        return
    
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
                
                context = neo4j_helper.query_dongyi_kg(user_query, limit=5)
                
                if not context:
                    print("❌ Không tìm thấy thông tin liên quan")
                    print("💡 Thử từ khóa: 'sốt', 'ho', 'đau đầu', 'lá tre', 'thạch cao', 'thanh nhiệt'\n")
                    continue
                
                if mode == "rag" and ollama_service:
                    print("=" * 70)
                    answer = ollama_service.generate_answer(user_query, context)
                    print(answer)
                    print("=" * 70)
                else:
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
    print("\n" + "="*70)
    print("📖 HƯỚNG DẪN SỬ DỤNG")
    print("="*70)
    print("• Nhập câu hỏi về Đông y để tìm kiếm bài thuốc")
    print("• 'help' - Xem hướng dẫn")
    print("• 'mode' - Chuyển đổi giữa chế độ RAG và RAW")
    print("• 'exit' - Thoát chương trình")
    print("\n💡 VÍ DỤ CÂU HỎI:")
    print("   - Bài thuốc chữa sốt cao")
    print("   - Thuốc nào có lá tre")
    print("   - Công hiệu thanh nhiệt")
    print("   - Chữa ho khát nước")
    print("   - Triệu chứng sốt buồn phiền")
    print("="*70 + "\n")


async def main():
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