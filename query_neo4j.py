"""
Hệ thống RAG 2 bước với LLM
Step 1: LLM phân tích câu hỏi → Tạo Cypher query
Step 2: LLM tổng hợp kết quả → Trả lời
"""

from neo4j import GraphDatabase
import requests
import json
import re
from typing import List, Dict, Optional

# --- Configuration ---
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "huy1552004"
NEO4J_DATABASE = "dongyi"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:latest"

print("=" * 70)
print("🔥 HỆ THỐNG RAG 2 BƯỚC - LLM-GUIDED QUERY")
print("=" * 70)
print(f"✓ Neo4j: {NEO4J_DATABASE}")
print(f"✓ Ollama: {OLLAMA_MODEL}")
print("=" * 70 + "\n")


class OllamaService:
    """Service gọi Ollama LLM"""
    
    def __init__(self, base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"
    
    def call_llm(self, prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> str:
        """Gọi LLM với prompt"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"❌ Lỗi Ollama: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"❌ Lỗi gọi LLM: {e}")
            return ""


class Neo4jRAG:
    """RAG System với LLM-guided query"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.database = NEO4J_DATABASE
        self.llm = OllamaService()
        
        # Schema Knowledge Graph
        self.kg_schema = """
NODES (Labels):
- BÀI_THUỐC: {tên_bài_thuốc, liều_lượng_cách_dùng, chú_ý, đối_tượng_phù_hợp}
- BỆNH: {tên_bệnh}
- TRIỆU_CHỨNG: {mô_tả}
- NGUYÊN_LIỆU: {tên_nguyên_liệu}
- CÔNG_HIỆU: {tên_công_hiệu}
- CÂY_THUỐC: {tên_chính, tên_khoa_học, tên_khác, họ, mô_tả, tính_vị_tác_dụng, công_dụng_chỉ_định, liều_dùng}
- HỌ_THỰC_VẬT: {tên_họ}
- BỘ_PHẬN_DÙNG: {tên_bộ_phận}
- THÀNH_PHẦN_HÓA_HỌC: {tên}

RELATIONSHIPS:
- (BÀI_THUỐC)-[:ĐIỀU_TRỊ]->(BỆNH)
- (BÀI_THUỐC)-[:TRỊ_TRIỆU_CHỨNG]->(TRIỆU_CHỨNG)
- (BÀI_THUỐC)-[:CHỨA_NGUYÊN_LIỆU]->(NGUYÊN_LIỆU)
- (BÀI_THUỐC)-[:CÓ_CÔNG_HIỆU]->(CÔNG_HIỆU)
- (NGUYÊN_LIỆU)-[:LÀ_DƯỢC_LIỆU_TỪ]->(CÂY_THUỐC)
- (CÂY_THUỐC)-[:THUỘC_HỌ]->(HỌ_THỰC_VẬT)
- (CÂY_THUỐC)-[:SỬ_DỤNG_BỘ_PHẬN]->(BỘ_PHẬN_DÙNG)
- (CÂY_THUỐC)-[:CHỨA_THÀNH_PHẦN]->(THÀNH_PHẦN_HÓA_HỌC)
"""
    
    def close(self):
        self.driver.close()
    
    # ========== STEP 1: LLM PHÂN TÍCH CÂU HỎI ==========
    def analyze_question_with_llm(self, question: str) -> Dict:
        """LLM phân tích câu hỏi - IMPROVED"""
        
        prompt = f"""Bạn là chuyên gia phân tích câu hỏi về Y học Đông y. 
Nhiệm vụ: Phân tích câu hỏi và trả về JSON với cấu trúc sau:

{{
  "intent": "<herb_info|disease_remedy|symptom_remedy|effect_info|remedy_list>",
  "main_entity": "<tên chính của thực thể cần tìm>",
  "entity_type": "<CÂY_THUỐC|BỆNH|TRIỆU_CHỨNG|BÀI_THUỐC|CÔNG_HIỆU>",
  "keywords": ["<keyword1>", "<keyword2>"],
  "search_target": "<mô tả ngắn gọn cần tìm gì>"
}}

ĐỊNH NGHĨA INTENT (ĐỌC KỸ):
- herb_info: Hỏi về thông tin CÂY THUỐC (tên, đặc điểm, công dụng, tính vị)
  VD: "cây đu đủ là gì?", "thông tin về bạc hà", "actisô có đặc điểm gì?"
  
- remedy_list: Hỏi về BÀI THUỐC liên quan đến một CÂY THUỐC cụ thể
  VD: "bài thuốc từ cây đu đủ", "bài thuốc có dây bói cá", "thuốc làm từ bạc hà"
  
- disease_remedy: Hỏi cách chữa BỆNH cụ thể
  VD: "thuốc gì chữa sốt?", "làm sao trị ho?", "chữa đau đầu bằng gì?"
  
- symptom_remedy: Hỏi cách chữa TRIỆU CHỨNG
  VD: "khó tiêu uống gì?", "mệt mỏi dùng thuốc gì?"
  
- effect_info: Hỏi về CÔNG HIỆU/TÁC DỤNG
  VD: "actisô có tác dụng gì?", "công dụng của lá tre?"

QUY TẮC PHÂN LOẠI:
1. Nếu câu hỏi có "bài thuốc" + "cây thuốc/tên cây" → intent = "remedy_list"
2. Nếu câu hỏi chỉ hỏi về cây thuốc → intent = "herb_info"
3. Nếu hỏi "chữa/trị" + tên bệnh → intent = "disease_remedy"

VÍ DỤ:
Câu hỏi: "cây đu đủ là gì?"
→ {{"intent": "herb_info", "main_entity": "đu đủ", "entity_type": "CÂY_THUỐC", "keywords": ["đu đủ"], "search_target": "thông tin về cây đu đủ"}}

Câu hỏi: "bài thuốc từ cây đu đủ"
→ {{"intent": "remedy_list", "main_entity": "đu đủ", "entity_type": "CÂY_THUỐC", "keywords": ["đu đủ"], "search_target": "các bài thuốc sử dụng cây đu đủ"}}

Câu hỏi: "thuốc gì chữa sốt cao?"
→ {{"intent": "disease_remedy", "main_entity": "sốt cao", "entity_type": "BỆNH", "keywords": ["sốt", "cao"], "search_target": "bài thuốc chữa sốt cao"}}

Câu hỏi: "actisô có tác dụng gì?"
→ {{"intent": "effect_info", "main_entity": "actisô", "entity_type": "CÂY_THUỐC", "keywords": ["actisô"], "search_target": "công dụng của actisô"}}

BÂY GIỜ PHÂN TÍCH:
Câu hỏi: "{question}"

CHỈ TRẢ VỀ JSON, KHÔNG GIẢI THÍCH THÊM:"""

        print("🤖 STEP 1: LLM đang phân tích câu hỏi...")
        
        response = self.llm.call_llm(prompt, temperature=0.1, max_tokens=300)
        
        try:
            # Extract JSON từ response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                print(f"   ✓ Intent: {analysis.get('intent')}")
                print(f"   ✓ Main entity: {analysis.get('main_entity')}")
                print(f"   ✓ Entity type: {analysis.get('entity_type')}")
                return analysis
            else:
                print(f"   ⚠️  Không parse được JSON: {response[:100]}")
                return self._fallback_analysis(question)
        except Exception as e:
            print(f"   ❌ Lỗi parse JSON: {e}")
            return self._fallback_analysis(question)
    
    def _fallback_analysis(self, question: str) -> Dict:
        """Fallback nếu LLM không trả về JSON - IMPROVED"""
        question_lower = question.lower()
        
        # Check "bài thuốc" + tên cây
        if 'bài thuốc' in question_lower and any(kw in question_lower for kw in ['cây ', 'từ ', 'có ']):
            # Extract tên cây
            for prefix in ['cây thuốc ', 'cây ', 'từ cây ', 'có ']:
                if prefix in question_lower:
                    parts = question_lower.split(prefix)
                    if len(parts) > 1:
                        herb_name = parts[1].strip()
                        return {
                            "intent": "remedy_list",
                            "main_entity": herb_name,
                            "entity_type": "CÂY_THUỐC",
                            "keywords": [herb_name],
                            "search_target": f"bài thuốc từ {herb_name}"
                        }
    
        # Check hỏi về cây thuốc
        if any(kw in question_lower for kw in ['cây ', 'thảo dược', 'dược liệu']):
            herb_name = question_lower.replace('cây ', '').replace('thảo dược ', '').strip()
            return {
                "intent": "herb_info",
                "main_entity": herb_name,
                "entity_type": "CÂY_THUỐC",
                "keywords": [herb_name],
                "search_target": "thông tin cây thuốc"
            }
        
        # Check chữa bệnh
        elif any(kw in question_lower for kw in ['chữa', 'trị', 'thuốc nào']):
            return {
                "intent": "disease_remedy",
                "main_entity": question.strip(),
                "entity_type": "BỆNH",
                "keywords": [question.strip()],
                "search_target": "bài thuốc chữa bệnh"
            }
        
        else:
            return {
                "intent": "general",
                "main_entity": question.strip(),
                "entity_type": "UNKNOWN",
                "keywords": [question.strip()],
                "search_target": "tìm kiếm chung"
            }
    
    # ========== STEP 2: TẠO VÀ CHẠY CYPHER QUERY ==========
    def generate_cypher_query(self, analysis: Dict) -> str:
        """Tạo Cypher query dựa trên phân tích"""
        
        intent = analysis.get('intent')
        main_entity = analysis.get('main_entity', '')
        keywords = analysis.get('keywords', [main_entity])
        
        print(f"\n🔍 STEP 2: Tạo Cypher query cho intent '{intent}'...")
        
        if intent == 'herb_info':
            # Tìm thông tin CÂY THUỐC - SỬA ĐỔI
            query = """
            MATCH (c:CÂY_THUỐC)
            WHERE ANY(kw IN $keywords WHERE 
                (c.tên_chính IS NOT NULL AND toLower(toString(c.tên_chính)) CONTAINS toLower(kw))
                OR (c.tên_khác IS NOT NULL AND toLower(toString(c.tên_khác)) CONTAINS toLower(kw))
                OR (c.tên_khoa_học IS NOT NULL AND toLower(toString(c.tên_khoa_học)) CONTAINS toLower(kw))
            )
            
            OPTIONAL MATCH (c)<-[:LÀ_DƯỢC_LIỆU_TỪ]-(n:NGUYÊN_LIỆU)<-[:CHỨA_NGUYÊN_LIỆU]-(r:BÀI_THUỐC)
            OPTIONAL MATCH (c)-[:THUỘC_HỌ]->(h:HỌ_THỰC_VẬT)
            
            RETURN 
                c.tên_chính AS ten_cay,
                c.tên_khoa_học AS ten_khoa_hoc,
                c.tên_khác AS ten_khac,
                c.họ AS ho,
                c.mô_tả AS mo_ta,
                c.tính_vị_tác_dụng AS tinh_vi,
                c.công_dụng_chỉ_định AS cong_dung,
                c.liều_dùng AS lieu_dung,
                collect(DISTINCT h.tên_họ)[0] AS ho_thuc_vat,
                collect(DISTINCT r.tên_bài_thuốc)[..3] AS bai_thuoc_lien_quan
            LIMIT 3
            """
            
        elif intent == 'remedy_list':
            # MỚI: Tìm BÀI THUỐC từ CÂY THUỐC
            query = """
            MATCH (c:CÂY_THUỐC)<-[:LÀ_DƯỢC_LIỆU_TỪ]-(n:NGUYÊN_LIỆU)<-[:CHỨA_NGUYÊN_LIỆU]-(r:BÀI_THUỐC)
            WHERE ANY(kw IN $keywords WHERE 
                c.tên_chính IS NOT NULL AND 
                toLower(toString(c.tên_chính)) CONTAINS toLower(kw)
            )
            
            OPTIONAL MATCH (r)-[:ĐIỀU_TRỊ]->(b:BỆNH)
            OPTIONAL MATCH (r)-[:CÓ_CÔNG_HIỆU]->(e:CÔNG_HIỆU)
            
            RETURN 
                c.tên_chính AS ten_cay_thuoc,
                r.tên_bài_thuốc AS ten_bai_thuoc,
                n.tên_nguyên_liệu AS nguyen_lieu,
                r.liều_lượng_cách_dùng AS lieu_luong,
                collect(DISTINCT b.tên_bệnh) AS benh_dieu_tri,
                collect(DISTINCT e.tên_công_hiệu) AS cong_hieu,
                r.chú_ý AS chu_y
            LIMIT 5
            """
            
        elif intent == 'disease_remedy' or intent == 'symptom_remedy':
            # Tìm BÀI THUỐC chữa BỆNH/TRIỆU CHỨNG - SỬA ĐỔI
            query = """
            MATCH (r:BÀI_THUỐC)
            WHERE EXISTS {
                MATCH (r)-[:ĐIỀU_TRỊ]->(b:BỆNH)
                WHERE ANY(kw IN $keywords WHERE 
                    b.tên_bệnh IS NOT NULL AND 
                    toLower(toString(b.tên_bệnh)) CONTAINS toLower(kw)
                )
            }
            OR EXISTS {
                MATCH (r)-[:TRỊ_TRIỆU_CHỨNG]->(s:TRIỆU_CHỨNG)
                WHERE ANY(kw IN $keywords WHERE 
                    s.mô_tả IS NOT NULL AND 
                    toLower(toString(s.mô_tả)) CONTAINS toLower(kw)
                )
            }
            
            OPTIONAL MATCH (r)-[:ĐIỀU_TRỊ]->(b:BỆNH)
            OPTIONAL MATCH (r)-[:CHỨA_NGUYÊN_LIỆU]->(n:NGUYÊN_LIỆU)
            OPTIONAL MATCH (r)-[:CÓ_CÔNG_HIỆU]->(e:CÔNG_HIỆU)
            
            RETURN 
                r.tên_bài_thuốc AS ten_bai_thuoc,
                collect(DISTINCT b.tên_bệnh) AS benh,
                collect(DISTINCT n.tên_nguyên_liệu) AS nguyen_lieu,
                collect(DISTINCT e.tên_công_hiệu) AS cong_hieu,
                r.liều_lượng_cách_dùng AS lieu_luong,
                r.chú_ý AS chu_y
            LIMIT 5
            """
            
        elif intent == 'effect_info':
            # Tìm CÔNG HIỆU - SỬA ĐỔI
            query = """
            MATCH (c:CÂY_THUỐC)
            WHERE ANY(kw IN $keywords WHERE 
                c.tên_chính IS NOT NULL AND 
                toLower(toString(c.tên_chính)) CONTAINS toLower(kw)
            )
            
            RETURN 
                c.tên_chính AS ten_cay,
                c.công_dụng_chỉ_định AS cong_dung,
                c.tính_vị_tác_dụng AS tinh_vi
            LIMIT 3
            """
            
        else:
            # General search - SỬA ĐỔI
            query = """
            MATCH (n)
            WHERE ANY(prop IN keys(n) WHERE 
                n[prop] IS NOT NULL AND 
                toLower(toString(n[prop])) CONTAINS toLower($keywords[0])
            )
            RETURN labels(n)[0] AS type, properties(n) AS props
            LIMIT 5
            """
    
        return query
    
    def execute_query(self, cypher: str, keywords: List[str]) -> List[Dict]:
        """Thực thi Cypher query"""
        try:
            with self.driver.session(database=self.database) as session:
                results = session.run(cypher, keywords=keywords)
                data = [dict(record) for record in results]
                print(f"   ✓ Tìm thấy {len(data)} kết quả")
                return data
        except Exception as e:
            print(f"   ❌ Lỗi query Neo4j: {e}")
            return []
    
    # ========== STEP 3: LLM TỔNG HỢP TRẢ LỜI ==========
    def generate_final_answer(self, question: str, analysis: Dict, kg_data: List[Dict]) -> str:
        """LLM tổng hợp dữ liệu từ KG và trả lời - IMPROVED"""
        
        if not kg_data:
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở tri thức."
        
        print(f"\n💬 STEP 3: LLM đang tổng hợp câu trả lời...")
        
        # Format dữ liệu KG
        kg_context = json.dumps(kg_data, ensure_ascii=False, indent=2)
        
        intent = analysis.get('intent')
        
        # Tùy chỉnh hướng dẫn theo intent
        if intent == 'remedy_list':
            instruction = """
- Liệt kê CÁC BÀI THUỐC sử dụng cây thuốc này
- Với mỗi bài thuốc, nêu: tên, liều lượng, bệnh điều trị, chú ý
- Trình bày dạng danh sách có số thứ tự
"""
        elif intent == 'herb_info':
            instruction = """
- Mô tả CÂY THUỐC: tên khoa học, họ thực vật, đặc điểm
- Nêu tính vị, công dụng chính
- Liệt kê một vài bài thuốc liên quan (nếu có)
"""
        elif intent == 'effect_info':
            instruction = """
- Tập trung vào CÔNG DỤNG, TÁC DỤNG của cây thuốc/thuốc
- Giải thích tính vị, công dụng chỉ định
- Liệt kê các công hiệu cụ thể
"""
        else:
            instruction = """
- Trả lời ĐÚNG TRỌNG TÂM câu hỏi
- Nêu rõ tên bài thuốc/cây thuốc, thành phần, công dụng
"""
        
        prompt = f"""Bạn là chuyên gia Y học Đông y Việt Nam. Dựa trên dữ liệu từ cơ sở tri thức, hãy trả lời câu hỏi một cách CHÍNH XÁC, NGẮN GỌN và DỄ HIỂU.

DỮ LIỆU TỪ KNOWLEDGE GRAPH:
{kg_context}

CÂU HỎI: {question}

HƯỚNG DẪN:
{instruction}
- KHÔNG bịa đặt thông tin không có trong dữ liệu
- Nếu có liều lượng, nêu rõ
- Kết thúc bằng lưu ý "nên tham khảo bác sĩ Đông y"
- TỐI ĐA 400 từ

TRẢ LỜI:"""

        answer = self.llm.call_llm(prompt, temperature=0.3, max_tokens=600)
        return answer
    
    # ========== MAIN RAG FLOW ==========
    def query(self, question: str) -> str:
        """Main RAG pipeline"""
        print("\n" + "=" * 70)
        print(f"❓ CÂU HỎI: {question}")
        print("=" * 70)
        
        # Step 1: LLM phân tích câu hỏi
        analysis = self.analyze_question_with_llm(question)
        
        # Step 2: Tạo và chạy Cypher query
        cypher = self.generate_cypher_query(analysis)
        kg_data = self.execute_query(cypher, analysis.get('keywords', [question]))
        
        # Step 3: LLM tổng hợp trả lời
        answer = self.generate_final_answer(question, analysis, kg_data)
        
        print("\n" + "=" * 70)
        print("📝 TRẢ LỜI:")
        print("=" * 70)
        print(answer)
        print("=" * 70 + "\n")
        
        return answer


# ========== INTERACTIVE MODE ==========
def main():
    print("\n🚀 KHỞI ĐỘNG HỆ THỐNG RAG 2 BƯỚC\n")
    
    rag = Neo4jRAG()
    
    print("💡 Nhập 'exit' để thoát, 'help' để xem ví dụ\n")
    
    try:
        while True:
            question = input("💬 Câu hỏi: ").strip()
            
            if not question:
                continue
            
            if question.lower() == 'exit':
                print("\n👋 Tạm biệt!\n")
                break
            
            if question.lower() == 'help':
                print("""
📖 VÍ DỤ CÂU HỎI:
  - Cây đu đủ là gì?
  - Actisô có tác dụng gì?
  - Thuốc gì chữa sốt cao?
  - Bài thuốc nào trị ho?
  - Lá tre có công dụng gì?
""")
                continue
            
            rag.query(question)
            
    except KeyboardInterrupt:
        print("\n\n👋 Tạm biệt!\n")
    finally:
        rag.close()


if __name__ == "__main__":
    main()