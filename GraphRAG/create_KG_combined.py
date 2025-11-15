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
    """Parse comma-separated or semicolon-separated list"""
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
    """Extract ingredients from dosage and usage instructions"""
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

def extract_effects(cong_hieu):
    """Extract individual effects from combined effects string"""
    if not isinstance(cong_hieu, str) or not cong_hieu.strip():
        return []
    
    effects = re.split(r'[,\.\;]', cong_hieu)
    return [uppercase_first_letter(effect.strip()) for effect in effects if effect.strip()]

# ============= PROCESS BÀI THUỐC =============
def process_bai_thuoc(row):
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

        # Disease nodes
        if chua_tri and isinstance(chua_tri, str) and chua_tri.strip():
            diseases = re.split(r'[,;]|\s+và\s+', chua_tri)
            for disease_name in diseases:
                disease_name = disease_name.strip()
                if disease_name:
                    disease_node = get_or_create_node(
                        "BỆNH", "tên_bệnh", uppercase_first_letter(disease_name)
                    )
                    graph.create(Relationship(remedy_node, "CHỮA TRỊ", disease_node))

        # Ingredient nodes
        ingredients = extract_ingredients(lieu_luong_cach_dung)
        for ingredient_info in ingredients:
            ingredient_node = get_or_create_node(
                "NGUYÊN LIỆU", "tên_nguyên_liệu", ingredient_info['name'],
                liều_lượng=ingredient_info['quantity']
            )
            # Tạo link với CÂY THUỐC nếu tồn tại
            herb_node = graph.nodes.match("CÂY THUỐC", tên_chính=ingredient_info['name']).first()
            if herb_node:
                graph.create(Relationship(ingredient_node, "LÀ", herb_node))
            
            graph.create(Relationship(remedy_node, "CHỨA NGUYÊN LIỆU", ingredient_node, liều_lượng=ingredient_info['quantity']))

        # Effect nodes
        effects = extract_effects(cong_hieu)
        for effect_name in effects:
            if effect_name:
                effect_node = get_or_create_node(
                    "CÔNG HIỆU", "tên_công_hiệu", effect_name
                )
                graph.create(Relationship(remedy_node, "CÓ CÔNG HIỆU", effect_node))

        print(f"✓ Đã xử lý bài thuốc: {ten_bai_thuoc}")

    except Exception as e:
        print(f"❌ Error processing bài thuốc: {e}")

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
        don_thuoc = row['Đơn thuốc']

        # Herb node
        if ten_chinh:
            herb_node = get_or_create_node(
                "CÂY THUỐC", "tên_chính", ten_chinh,
                tên_khoa_học=ten_khoa_hoc if isinstance(ten_khoa_hoc, str) else "",
                mô_tả=mo_ta if isinstance(mo_ta, str) else "",
                nơi_sống_thu_hái=noi_song_thu_hai if isinstance(noi_song_thu_hai, str) else "",
                thành_phần_hóa_học=thanh_phan_hoa_hoc if isinstance(thanh_phan_hoa_hoc, str) else "",
                tính_vị_tác_dụng=tinh_vi_tac_dung if isinstance(tinh_vi_tac_dung, str) else "",
                liều_dùng=lieu_dung if isinstance(lieu_dung, str) else ""
            )

            # Aliases
            ten_khac_list = parse_list_field(ten_khac)
            for alias in ten_khac_list:
                alias_node = get_or_create_node("TÊN KHÁC", "tên", alias)
                graph.create(Relationship(herb_node, "CÓ TÊN GỌI KHÁC", alias_node))

            # Family
            if ho and isinstance(ho, str) and ho.strip():
                family_node = get_or_create_node("HỌ THỰC VẬT", "tên_họ", uppercase_first_letter(ho))
                graph.create(Relationship(herb_node, "THUỘC HỌ", family_node))

            # Parts used
            bo_phan_list = parse_list_field(bo_phan_dung)
            for bo_phan in bo_phan_list:
                part_node = get_or_create_node("BỘ PHẬN DÙNG", "tên_bộ_phận", bo_phan)
                graph.create(Relationship(herb_node, "SỬ DỤNG BỘ PHẬN", part_node))

            # Uses
            if cong_dung_chi_dinh and isinstance(cong_dung_chi_dinh, str):
                cong_dung_list = re.split(r'[,;]|\s+và\s+', cong_dung_chi_dinh)
                for cong_dung in cong_dung_list:
                    cong_dung = cong_dung.strip()
                    if cong_dung:
                        use_node = get_or_create_node("CÔNG DỤNG", "mô_tả", uppercase_first_letter(cong_dung))
                        graph.create(Relationship(herb_node, "CÓ CÔNG DỤNG", use_node))

            # Prescriptions
            if don_thuoc and isinstance(don_thuoc, str) and don_thuoc.strip():
                prescription_node = get_or_create_node(
                    "ĐƠN THUỐC", "mô_tả", don_thuoc.strip()
                )
                graph.create(Relationship(herb_node, "DÙNG TRONG ĐƠN", prescription_node))

            # Chemical components
            thanh_phan_list = parse_list_field(thanh_phan_hoa_hoc)
            for thanh_phan in thanh_phan_list:
                if len(thanh_phan) > 3:
                    chemical_node = get_or_create_node("THÀNH PHẦN HÓA HỌC", "tên", thanh_phan)
                    graph.create(Relationship(herb_node, "CHỨA THÀNH PHẦN", chemical_node))

            print(f"✓ Đã xử lý cây thuốc: {ten_chinh}")

    except Exception as e:
        print(f"❌ Error processing cây thuốc: {e}")

if __name__ == "__main__":
    # Connect to Neo4j
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "huy1552004"
    
    graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), name="yhct")
    
    print("=" * 60)
    print("🚀 BẮT ĐẦU TẠO KNOWLEDGE GRAPH KẾT HỢP")
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
    print("✅ HOÀN THÀNH TẠO KNOWLEDGE GRAPH KẾT HỢP!")
    print("=" * 60)
    print(f"\n📊 Thống kê:")
    print(f"   - Cây thuốc: {len(df_cay_thuoc)}")
    print(f"   - Bài thuốc: {len(df_bai_thuoc)}")
    print(f"\n🌐 Xem trong Neo4j Browser: http://localhost:7474")
    print(f"   Database: dongyi_combined")