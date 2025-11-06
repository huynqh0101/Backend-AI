from py2neo import Graph, Node, Relationship
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

def uppercase_first_letter(text):
    if isinstance(text, str) and text.strip():
        return text.strip().capitalize()
    else:
        return text
    
def clear_graph():
    query = """
    MATCH (n)
    DETACH DELETE n
    """
    graph.run(query)
    print("Graph has been cleared...")

def extract_ingredients(lieu_luong_cach_dung):
    """Extract ingredients from dosage and usage instructions"""
    if not isinstance(lieu_luong_cach_dung, str) or not lieu_luong_cach_dung.strip():
        return []
    
    # Pattern to extract ingredients with quantities (e.g., "200g lá tre tươi", "100g thạch cao")
    pattern = r'(\d+(?:\.\d+)?g?\s*(?:ml)?)\s+([^,\.]+?)(?=\s*[,\.]|\s+\d+|\s*$)'
    matches = re.findall(pattern, lieu_luong_cach_dung)
    
    ingredients = []
    for quantity, ingredient in matches:
        ingredient = ingredient.strip()
        if len(ingredient) > 2:  # Filter out very short matches
            ingredients.append({
                'name': uppercase_first_letter(ingredient),
                'quantity': quantity.strip()
            })
    
    return ingredients

def extract_effects(cong_hieu):
    """Extract individual effects from combined effects string"""
    if not isinstance(cong_hieu, str) or not cong_hieu.strip():
        return []
    
    # Split by common delimiters
    effects = re.split(r'[,\.\;]', cong_hieu)
    return [uppercase_first_letter(effect.strip()) for effect in effects if effect.strip()]

def get_or_create_node(label, key, value, **properties):
    node = graph.nodes.match(label, **{key: value}).first()
    if node is None:
        node = Node(label, **{key: value}, **properties)
        graph.create(node)
    return node

def process_row(row):
    try:
        chuong_so = row['chuong_so']
        tieu_de_chuong = row['tieu_de_chuong']
        ten_bai_thuoc = row['ten_bai_thuoc']
        chua_tri = row['chua_tri']
        lieu_luong_cach_dung = row['lieu_luong_cach_dung']
        cong_hieu = row['cong_hieu']
        chu_y = row['chu_y']
        ghi_chu = row['ghi_chu']
        doi_tuong_phu_hop = row['doi_tuong_phu_hop']
        luu_y = row['luu_y']
        cong_dung_khac = row['cong_dung_khac']

        # Chapter node
        if chuong_so and tieu_de_chuong:
            chapter_node = get_or_create_node(
                "CHƯƠNG", "số_chương", int(chuong_so),
                tiêu_đề=tieu_de_chuong
            )

        # Remedy node
        if ten_bai_thuoc:
            remedy_node = get_or_create_node(
                "BÀI THUỐC", "tên_bài_thuốc", ten_bai_thuoc,
                liều_lượng_cách_dùng=lieu_luong_cach_dung if isinstance(lieu_luong_cach_dung, str) else "",
                chú_ý=chu_y if isinstance(chu_y, str) else "",
                ghi_chú=ghi_chu if isinstance(ghi_chu, str) else "",
                đối_tượng_phù_hợp=doi_tuong_phu_hop if isinstance(doi_tuong_phu_hop, str) else "",
                lưu_ý=luu_y if isinstance(luu_y, str) else "",
                công_dụng_khác=cong_dung_khac if isinstance(cong_dung_khac, str) else ""
            )
            if chuong_so and tieu_de_chuong:
                graph.create(Relationship(chapter_node, "CHỨA", remedy_node))

        # Disease/Condition nodes
        if chua_tri and isinstance(chua_tri, str) and chua_tri.strip():
            print(f"🔍 DEBUG - Bài thuốc: {ten_bai_thuoc}")
            print(f"   Chữa trị: '{chua_tri}'")
            diseases = re.split(r'[,;]|\s+và\s+', chua_tri)
            print(f"   Số bệnh tìm thấy: {len(diseases)}")
            for disease_name in diseases:
                disease_name = disease_name.strip()
                if disease_name:
                    print(f"   ✓ Tạo bệnh: '{disease_name}'")
                    disease_node = get_or_create_node(
                        "BỆNH", "tên_bệnh", uppercase_first_letter(disease_name)
                    )
                    graph.create(Relationship(remedy_node, "CHỮA TRỊ", disease_node))
        else:
            print(f"⚠️  Bỏ qua bài thuốc '{ten_bai_thuoc}' - không có thông tin chữa trị")

        # Ingredient nodes
        ingredients = extract_ingredients(lieu_luong_cach_dung)
        for ingredient_info in ingredients:
            ingredient_node = get_or_create_node(
                "NGUYÊN LIỆU", "tên_nguyên_liệu", ingredient_info['name'],
                liều_lượng=ingredient_info['quantity']
            )
            graph.create(Relationship(remedy_node, "CHỨA NGUYÊN LIỆU", ingredient_node, liều_lượng=ingredient_info['quantity']))

        # Effect nodes
        effects = extract_effects(cong_hieu)
        for effect_name in effects:
            if effect_name:
                effect_node = get_or_create_node(
                    "CÔNG HIỆU", "tên_công_hiệu", effect_name
                )
                graph.create(Relationship(remedy_node, "CÓ CÔNG HIỆU", effect_node))

    except Exception as e:
        print(f"Error processing row: {e}")
        print(f"Row data: {row}")

if __name__ == "__main__": 
    # Connect to Neo4j
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "huy1552004"
    
    graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), name="dongyi")
    
    print("Clearing existing graph...")
    clear_graph()
    
    print("Loading data from CSV...")
    # Thêm các tham số để xử lý file CSV có lỗi
    df_cn = pd.read_csv(
        r'.\data\data_translated.csv', 
        encoding="utf-8",
        on_bad_lines='skip',  # Bỏ qua các dòng lỗi
        engine='python'       # Sử dụng engine Python linh hoạt hơn
    )
    
    print(f"Successfully loaded {len(df_cn)} rows")
    print(f"Processing {len(df_cn)} rows...")
    num_workers = 4
    
    # Process each row in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_row, row) for index, row in df_cn.iterrows()]
        for i, future in enumerate(as_completed(futures)):
            try:
                future.result()
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(df_cn)} rows...")
            except Exception as e:
                print(f"Error processing row: {e}")
    
    print("Knowledge Graph creation completed!")