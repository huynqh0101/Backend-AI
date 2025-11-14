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

def process_row(row):
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

        # 1. Tạo node CÂY THUỐC chính
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

            # 2. Tạo node TÊN KHÁC (aliases)
            ten_khac_list = parse_list_field(ten_khac)
            for alias in ten_khac_list:
                alias_node = get_or_create_node("TÊN KHÁC", "tên", alias)
                graph.create(Relationship(herb_node, "CÓ TÊN GỌI KHÁC", alias_node))

            # 3. Tạo node HỌ (Family)
            if ho and isinstance(ho, str) and ho.strip():
                family_node = get_or_create_node("HỌ THỰC VẬT", "tên_họ", uppercase_first_letter(ho))
                graph.create(Relationship(herb_node, "THUỘC HỌ", family_node))

            # 4. Tạo node BỘ PHẬN DÙNG
            bo_phan_list = parse_list_field(bo_phan_dung)
            for bo_phan in bo_phan_list:
                part_node = get_or_create_node("BỘ PHẬN DÙNG", "tên_bộ_phận", bo_phan)
                graph.create(Relationship(herb_node, "SỬ DỤNG BỘ PHẬN", part_node))

            # 5. Tạo node CÔNG DỤNG (từ cột "Công dụng, chỉ định và phối hợp")
            if cong_dung_chi_dinh and isinstance(cong_dung_chi_dinh, str):
                # Tách ra các công dụng riêng lẻ
                cong_dung_list = re.split(r'[,;]|\s+và\s+', cong_dung_chi_dinh)
                for cong_dung in cong_dung_list:
                    cong_dung = cong_dung.strip()
                    if cong_dung:
                        use_node = get_or_create_node("CÔNG DỤNG", "mô_tả", uppercase_first_letter(cong_dung))
                        graph.create(Relationship(herb_node, "CÓ CÔNG DỤNG", use_node))

            # 6. Tạo node ĐƠN THUỐC (nếu có)
            if don_thuoc and isinstance(don_thuoc, str) and don_thuoc.strip():
                prescription_node = get_or_create_node(
                    "ĐƠN THUỐC", "mô_tả", don_thuoc.strip()
                )
                graph.create(Relationship(herb_node, "DÙNG TRONG ĐƠN", prescription_node))

            # 7. Tạo node THÀNH PHẦN HÓA HỌC (nếu muốn chi tiết)
            thanh_phan_list = parse_list_field(thanh_phan_hoa_hoc)
            for thanh_phan in thanh_phan_list:
                if len(thanh_phan) > 3:  # Lọc các thành phần quá ngắn
                    chemical_node = get_or_create_node("THÀNH PHẦN HÓA HỌC", "tên", thanh_phan)
                    graph.create(Relationship(herb_node, "CHỨA THÀNH PHẦN", chemical_node))

            print(f"✓ Đã tạo node cho cây thuốc: {ten_chinh}")

    except Exception as e:
        print(f"❌ Error processing row: {e}")
        print(f"Row data: {row}")

if __name__ == "__main__": 
    # Connect to Neo4j
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "huy1552004"
    
    graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), name="cay_thuoc_db")
    
    print("Clearing existing graph...")
    clear_graph()
    
    print("Loading data from CSV...")
    # Đổi đường dẫn file CSV của bạn tại đây
    df = pd.read_csv(
        r'.\data\cay_thuoc.csv',  # ĐỔI TÊN FILE CỦA BẠN
        encoding="utf-8",
        on_bad_lines='skip',
        engine='python'
    )
    
    print(f"Successfully loaded {len(df)} rows")
    print(f"Processing {len(df)} rows...")
    num_workers = 4
    
    # Process each row in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_row, row) for index, row in df.iterrows()]
        for i, future in enumerate(as_completed(futures)):
            try:
                future.result()
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(df)} rows...")
            except Exception as e:
                print(f"Error processing row: {e}")
    
    print("✅ Knowledge Graph creation completed!")