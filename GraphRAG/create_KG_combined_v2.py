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

def parse_list_field(text):
    if not isinstance(text, str) or not text.strip():
        return []
    items = re.split(r'[,;]', text)
    return [uppercase_first_letter(item.strip()) for item in items if item.strip()]

def get_or_create_node(label, key, value, **properties):
    node = graph.nodes.match(label, **{key: value}).first()
    if node is None:
        node = Node(label, **{key: value}, **properties)
        graph.create(node)
    return node

def extract_ingredients(lieu_luong_cach_dung):
    if not isinstance(lieu_luong_cach_dung, str) or not lieu_luong_cach_dung.strip():
        return []
    
    pattern = r'(\d+(?:\.\d+)?g?\s*(?:ml)?)\s+([^,\.]+?)(?=\s*[,\.]|\s+\d+|\s*$)'
    matches = re.findall(pattern, lieu_luong_cach_dung)
    
    ingredients = []
    for quantity, ingredient in matches:
        ingredient = ingredient.strip()
        if len(ingredient) > 2:
            ingredients.append({
                'name': uppercase_first_letter(ingredient),
                'quantity': quantity.strip()
            })
    
    return ingredients

def extract_effects(text):
    if not isinstance(text, str) or not text.strip():
        return []
    
    effects = re.split(r'[,\.\;]', text)
    return [uppercase_first_letter(effect.strip()) for effect in effects if effect.strip()]

# ============= PROCESS BÀI THUỐC =============
def process_bai_thuoc(row):
    try:
        ten_bai_thuoc = row['ten_bai_thuoc']
        chua_tri = row['chua_tri']
        lieu_luong_cach_dung = row['lieu_luong_cach_dung']
        cong_hieu = row['cong_hieu']
        chu_y = row['chu_y']
        ghi_chu = row['ghi_chu']
        doi_tuong_phu_hop = row['doi_tuong_phu_hop']
        luu_y = row['luu_y']
        cong_dung_khac = row['cong_dung_khac']

        # 1. Tạo node BÀI THUỐC
        if ten_bai_thuoc:
            remedy_node = get_or_create_node(
                "Bài_Thuốc", "tên", ten_bai_thuoc,
                cách_dùng=lieu_luong_cach_dung if isinstance(lieu_luong_cach_dung, str) else "",
                chú_ý=chu_y if isinstance(chu_y, str) else "",
                ghi_chú=ghi_chu if isinstance(ghi_chu, str) else "",
                đối_tượng=doi_tuong_phu_hop if isinstance(doi_tuong_phu_hop, str) else "",
                lưu_ý=luu_y if isinstance(luu_y, str) else ""
            )

            # 2. Tạo node BỆNH và liên kết
            if chua_tri and isinstance(chua_tri, str) and chua_tri.strip():
                diseases = re.split(r'[,;]|\s+và\s+', chua_tri)
                for disease_name in diseases:
                    disease_name = disease_name.strip()
                    if disease_name:
                        disease_node = get_or_create_node(
                            "Bệnh", "tên", uppercase_first_letter(disease_name)
                        )
                        graph.create(Relationship(remedy_node, "CHỮA", disease_node))

            # 3. Tạo node NGUYÊN LIỆU và liên kết với CÂY THUỐC
            ingredients = extract_ingredients(lieu_luong_cach_dung)
            for ingredient_info in ingredients:
                ingredient_node = get_or_create_node(
                    "Nguyên_Liệu", "tên", ingredient_info['name']
                )
                
                # Link BÀI THUỐC -> NGUYÊN LIỆU
                rel = Relationship(remedy_node, "DÙNG", ingredient_node)
                rel['liều_lượng'] = ingredient_info['quantity']
                graph.create(rel)
                
                # Link NGUYÊN LIỆU -> CÂY THUỐC (nếu có)
                herb_node = graph.nodes.match("Cây_Thuốc", tên=ingredient_info['name']).first()
                if not herb_node:
                    # Thử tìm theo tên khác
                    alias_node = graph.nodes.match("Biệt_Danh", tên=ingredient_info['name']).first()
                    if alias_node:
                        herb_node = list(graph.match((None, None), r_type="GỌI_LÀ"))[0].start_node
                
                if herb_node:
                    graph.create(Relationship(ingredient_node, "LÀ", herb_node))

            # 4. Tạo node TÁC DỤNG (merge CÔNG HIỆU)
            effects = extract_effects(cong_hieu)
            for effect_name in effects:
                if effect_name:
                    effect_node = get_or_create_node("Tác_Dụng", "mô_tả", effect_name)
                    graph.create(Relationship(remedy_node, "CÓ", effect_node))

        print(f"✓ Đã xử lý bài thuốc: {ten_bai_thuoc}")

    except Exception as e:
        print(f"❌ Error processing bài thuốc: {e}")
        print(f"   Row: {row.get('ten_bai_thuoc', 'unknown')}")

# ============= PROCESS CÂY THUỐC =============
def process_cay_thuoc(row):
    try:
        ten_chinh = row['Tên chính']
        ten_khac = row['Tên khác']
        ten_khoa_hoc = row['Tên khoa học']
        ho = row['Họ']
        mo_ta = row['Mô tả']
        bo_phan_dung = row['Bộ phận dùng']
        noi_song_thu_hai = row['Nơi sống và thu hái']
        thanh_phan_hoa_hoc = row['Thành phần hoá học']
        tinh_vi_tac_dung = row['Tính vị, tác dụng']
        cong_dung_chi_dinh = row['Công dụng, chỉ định và phối hợp']
        lieu_dung = row['Liều dùng']

        # 1. Tạo node CÂY THUỐC
        if ten_chinh:
            herb_node = get_or_create_node(
                "Cây_Thuốc", "tên", ten_chinh,
                tên_khoa_học=ten_khoa_hoc if isinstance(ten_khoa_hoc, str) else "",
                mô_tả=mo_ta if isinstance(mo_ta, str) else "",
                nơi_sống=noi_song_thu_hai if isinstance(noi_song_thu_hai, str) else "",
                tính_vị=tinh_vi_tac_dung if isinstance(tinh_vi_tac_dung, str) else "",
                liều_dùng=lieu_dung if isinstance(lieu_dung, str) else ""
            )

            # 2. Tạo node BIỆT DANH (tên khác)
            ten_khac_list = parse_list_field(ten_khac)
            for alias in ten_khac_list:
                alias_node = get_or_create_node("Biệt_Danh", "tên", alias)
                graph.create(Relationship(herb_node, "GỌI_LÀ", alias_node))

            # 3. Tạo node HỌ
            if ho and isinstance(ho, str) and ho.strip():
                family_node = get_or_create_node("Họ", "tên", uppercase_first_letter(ho))
                graph.create(Relationship(herb_node, "THUỘC", family_node))

            # 4. Tạo node BỘ PHẬN
            bo_phan_list = parse_list_field(bo_phan_dung)
            for bo_phan in bo_phan_list:
                part_node = get_or_create_node("Bộ_Phận", "tên", bo_phan)
                graph.create(Relationship(herb_node, "DÙNG_PHẦN", part_node))

            # 5. Tạo node TÁC DỤNG (merge với CÔNG HIỆU từ bài thuốc)
            if cong_dung_chi_dinh and isinstance(cong_dung_chi_dinh, str):
                tac_dung_list = re.split(r'[,;]|\s+và\s+', cong_dung_chi_dinh)
                for tac_dung in tac_dung_list:
                    tac_dung = tac_dung.strip()
                    if tac_dung:
                        effect_node = get_or_create_node("Tác_Dụng", "mô_tả", uppercase_first_letter(tac_dung))
                        graph.create(Relationship(herb_node, "CÓ", effect_node))

            # 6. Tạo node HÓA CHẤT
            thanh_phan_list = parse_list_field(thanh_phan_hoa_hoc)
            for thanh_phan in thanh_phan_list:
                if len(thanh_phan) > 3:
                    chemical_node = get_or_create_node("Hóa_Chất", "tên", thanh_phan)
                    graph.create(Relationship(herb_node, "CHỨA", chemical_node))

            print(f"✓ Đã xử lý cây thuốc: {ten_chinh}")

    except Exception as e:
        print(f"❌ Error processing cây thuốc: {e}")
        print(f"   Row: {row.get('Tên chính', 'unknown')}")

if __name__ == "__main__":
    # Connect to Neo4j
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "huy1552004"
    
    graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), name="dongyi")
    
    print("=" * 60)
    print("🚀 TẠO KNOWLEDGE GRAPH KẾT HỢP (PHIÊN BẢN TỐI ƯU)")
    print("=" * 60)
    
    print("\n🗑️  Clearing existing graph...")
    clear_graph()
    
    # ============= LOAD CÂY THUỐC =============
    print("\n📚 PHẦN 1: LOADING CÂY THUỐC...")
    df_cay_thuoc = pd.read_csv(
        r'.\data\cay_thuoc.csv',
        encoding="utf-8",
        on_bad_lines='skip',
        engine='python'
    )
    print(f"✓ Loaded {len(df_cay_thuoc)} cây thuốc")
    
    print("\n⚙️  Processing cây thuốc...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_cay_thuoc, row) for _, row in df_cay_thuoc.iterrows()]
        for i, future in enumerate(as_completed(futures)):
            try:
                future.result()
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(df_cay_thuoc)} cây thuốc...")
            except Exception as e:
                print(f"   Error: {e}")
    
    # ============= LOAD BÀI THUỐC =============
    print("\n📚 PHẦN 2: LOADING BÀI THUỐC...")
    df_bai_thuoc = pd.read_csv(
        r'.\data\data_translated.csv',
        encoding="utf-8",
        on_bad_lines='skip',
        engine='python'
    )
    print(f"✓ Loaded {len(df_bai_thuoc)} bài thuốc")
    
    print("\n⚙️  Processing bài thuốc...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_bai_thuoc, row) for _, row in df_bai_thuoc.iterrows()]
        for i, future in enumerate(as_completed(futures)):
            try:
                future.result()
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(df_bai_thuoc)} bài thuốc...")
            except Exception as e:
                print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ HOÀN THÀNH TẠO KNOWLEDGE GRAPH!")
    print("=" * 60)
    print(f"\n📊 Thống kê:")
    print(f"   - Cây thuốc: {len(df_cay_thuoc)}")
    print(f"   - Bài thuốc: {len(df_bai_thuoc)}")
    print(f"\n🌐 Xem trong Neo4j Browser: http://localhost:7474")