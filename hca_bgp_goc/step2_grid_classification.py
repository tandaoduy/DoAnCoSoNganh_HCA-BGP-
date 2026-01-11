"""
HCA-BGP Step 2: Grid Classification (Static Grid Only)
Input: Kết quả từ Step 1 (M, R) và dữ liệu điểm
Output: Các lưới được phân loại (core, dense, sparse, empty)
Chú ý: Chỉ phân loại lưới MxM tĩnh, CHƯA chia đệ quy
"""
"""Step 2: static grid construction and grid classification helpers.

Provides routines to build a static MxM grid, classify cells (core/dense/...
and plotting utilities used by the pipeline.
"""

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from grid_common import (
    diagonal_length,
    compute_centroid,
    compute_pj_for_cell,
    compute_Dj_for_cell,
)


# =============================
# LỚP BIỂU DIỄN Ô LƯỚI (GridCell)
# =============================
class GridCell:
    """Đại diện cho một ô lưới trong grid MxM.

    Thay vì dùng dict rời rạc, ta gom toàn bộ thông tin và
    các phép tính liên quan đến một ô lưới vào class này.
    Logic tính toán giữ nguyên so với phiên bản dùng dict.
    """

    def __init__(self, ix, iy, xmin, xmax, ymin, ymax):
        # Chỉ số ô trong lưới (tọa độ ô theo hàng/cột)
        self.ix = ix
        self.iy = iy

        # Biên hình học của ô trên mặt phẳng dữ liệu
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        # Danh sách điểm (x, y) thuộc ô này
        self.points = []

        # Loại ô: 'empty', 'sparse', 'dense', 'core'
        self.grid_type = None

    # -------------------------
    # CÁC PHƯƠNG THỨC TIỆN ÍCH
    # -------------------------
    def add_point(self, point):
        """Thêm một điểm (x, y) vào ô lưới."""
        self.points.append(point)

    def count(self):
        """Trả về số điểm hiện đang nằm trong ô."""
        return len(self.points)

    def center(self):
        """Tính tâm hình học của ô lưới (Gj - Equation 5)."""
        center_x = (self.xmin + self.xmax) / 2.0
        center_y = (self.ymin + self.ymax) / 2.0
        return (center_x, center_y)

    def centroid(self):
        """Tính trọng tâm các điểm trong ô (Cj - Equation 4)."""
        return compute_centroid(self.points)

    def diagonal_length(self):
        """Chiều dài đường chéo của ô lưới, dùng cho các phép chuẩn hóa."""
        return diagonal_length(self.xmin, self.xmax, self.ymin, self.ymax)

    def compute_pj(self):
        """Tính pj - độ lệch giữa tâm lưới và trọng tâm điểm (Equation 6).

        pj = ||Gj - Cj|| / ||GMAX_j - GMIN_j||
        """
        return compute_pj_for_cell(
            self.points,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
        )

    def compute_Dj(self):
        """Tính Dj - độ phân tán điểm trong ô (Equation 7).

        Dj = std_distance / diagonal
        """
        return compute_Dj_for_cell(
            self.points,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
        )


# =============================
# HÀM THUẬN TIỆN BAO QUANH GridCell
# =============================
def create_grid_cell(ix, iy, xmin, xmax, ymin, ymax):
    """Tạo một ô lưới GridCell (thay cho dict phiên bản cũ)."""
    return GridCell(ix, iy, xmin, xmax, ymin, ymax)


def gridcell_add_point(cell, p):
    """Thêm điểm vào ô (hàm bao quanh GridCell.add_point)."""
    cell.add_point(p)


def gridcell_count(cell):
    """Đếm số điểm trong ô (hàm bao quanh GridCell.count)."""
    return cell.count()


def gridcell_center(cell):
    """Tâm hình học của ô lưới (Gj - Equation 5)."""
    return cell.center()


def gridcell_centroid(cell):
    """Trọng tâm các điểm trong ô (Cj - Equation 4)."""
    return cell.centroid()


def gridcell_diagonal_length(cell):
    """Chiều dài đường chéo của ô lưới, dùng chung cho các phép tính chuẩn hóa."""
    return cell.diagonal_length()


def gridcell_compute_pj(cell):
    """Tính pj - độ lệch giữa tâm lưới và trọng tâm điểm (Equation 6)."""
    return cell.compute_pj()


def gridcell_compute_Dj(cell):
    """Tính Dj - độ phân tán điểm trong ô (Equation 7)."""
    return cell.compute_Dj()


# =============================
# XÂY DỰNG LƯỚI MxM
# =============================
def build_grid(points, M):
    """
    Tạo lưới MxM tĩnh và gán điểm vào các ô

    Returns:
        grid: dict {(ix, iy): cell_dict}
        bounds: (xmin, xmax, ymin, ymax)
    """
    if not points:
        raise ValueError("build_grid: 'points' must be a non-empty list of (x, y) tuples")

    if not isinstance(M, int) or M <= 0:
        raise ValueError("build_grid: 'M' must be a positive integer")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # Tránh chia cho 0
    if xmax == xmin:
        xmax += 1e-9
    if ymax == ymin:
        ymax += 1e-9

    grid = {}

    # Tạo MxM ô lưới
    for iy in range(M):  # Hàng (Y)
        for ix in range(M):  # Cột (X)
            x0 = xmin + (xmax - xmin) * (ix / M)
            x1 = xmin + (xmax - xmin) * ((ix + 1) / M)
            y0 = ymin + (ymax - ymin) * (iy / M)
            y1 = ymin + (ymax - ymin) * ((iy + 1) / M)

            grid[(ix, iy)] = create_grid_cell(ix, iy, x0, x1, y0, y1)

    # Gán điểm vào các ô
    for p in points:
        x, y = p
        ix = int(min(M - 1, max(0, math.floor((x - xmin) / (xmax - xmin) * M))))
        iy = int(min(M - 1, max(0, math.floor((y - ymin) / (ymax - ymin) * M))))
        gridcell_add_point(grid[(ix, iy)], p)

    return grid, (xmin, xmax, ymin, ymax)


# =============================
# PHÂN LOẠI LƯỚI
# =============================
def classify_grids(grid, R, pj_threshold=0.1, Dj_threshold=0.5):
    """
    Phân loại các ô lưới theo tiêu chí trong paper

    Tiêu chí:
    1. Empty grid: count = 0
    2. Sparse grid: count ≤ R
    3. Dense grid: count > R
       3a. Core grid: pj < 0.1 AND Dj < 0.5

       3b. Non-core dense: còn lại

    Returns:
        Các dict chứa các ô theo loại
    """
    empty_cells = []
    sparse_cells = []
    dense_cells = []
    core_cells = []

    for key, cell in grid.items():
        cnt = gridcell_count(cell)

        # 1. Ô rỗng
        if cnt == 0:
            cell.grid_type = 'empty'

            empty_cells.append(cell)
            continue

        # 2. Ô thưa
        if cnt <= R:
            cell.grid_type = 'sparse'

            sparse_cells.append(cell)
            continue

        # 3. Ô dày đặc - kiểm tra xem có phải core không
        pj = gridcell_compute_pj(cell)
        Dj = gridcell_compute_Dj(cell)

        if pj is not None and Dj is not None:
            if pj < pj_threshold and Dj < Dj_threshold:
                cell.grid_type = 'core'

                core_cells.append(cell)
            else:
                cell.grid_type = 'dense'

                dense_cells.append(cell)
        else:
            # Nếu không tính được pj, Dj → coi như dense
            cell.grid_type = 'dense'

            dense_cells.append(cell)

    return {
        'empty': empty_cells,
        'sparse': sparse_cells,
        'dense': dense_cells,
        'core': core_cells
    }


# =============================
# IN THỐNG KÊ
# =============================
def print_statistics(classified, R):
    """In thống kê chi tiết về các loại ô"""
    print("KẾT QUẢ PHÂN LOẠI LƯỚI")

    print(f"\n Ngưỡng R = {R:.4f}")
    print(f"Tổng số ô: {sum(len(v) for v in classified.values())}")
    print()

    # Thống kê từng loại
    print(f"⬜ Empty grids:    {len(classified['empty']):3d} ô (không có điểm)")
    print(f"🔵 Sparse grids:   {len(classified['sparse']):3d} ô (count ≤ R)")
    print(f"🟡 Dense grids:    {len(classified['dense']):3d} ô (count > R, không core)")
    print(f"🟢 Core grids:     {len(classified['core']):3d} ô (count > R, pj<0.1, Dj<0.5)")

    # Chi tiết core grids
    if classified['core']:
        print("\n" + "="*50)
        print("CHI TIẾT CORE GRIDS")
        print("="*50)
        for i, cell in enumerate(classified['core'][:10], 1):
            pj = gridcell_compute_pj(cell)
            Dj = gridcell_compute_Dj(cell)
            print(
                f"Core {i:2d}: Vị trí ({cell.ix}, {cell.iy}) | "
                f"Số điểm: {gridcell_count(cell):3d} | "
                f"pj={pj:.4f} | Dj={Dj:.4f}"
            )

        if len(classified['core']) > 10:
            print(f"... và {len(classified['core']) - 10} core grids khác")


# =============================
# IN CHI TIẾT TỪNG BƯỚC TÍNH TOÁN
# =============================
def print_detailed_calculation(grid, R, pj_threshold=0.1, Dj_threshold=0.5):
    """
    In chi tiết từng bước tính toán Cj, Gj, pj, Dj cho từng ô lưới không rỗng
    với công thức và kết quả cụ thể.
    """
    print("\n" + "="*80)
    print("CHI TIẾT TÍNH TOÁN PHÂN LOẠI LƯỚI")
    print("="*80)
    print(f"\n📌 Ngưỡng mật độ R = {R}")
    print(f"📌 Ngưỡng pj < {pj_threshold} (độ lệch tâm)")
    print(f"📌 Ngưỡng Dj < {Dj_threshold} (độ phân tán)")
    
    # Lọc các ô không rỗng và sắp xếp theo vị trí
    non_empty_cells = [(key, cell) for key, cell in grid.items() if cell.count() > 0]
    non_empty_cells.sort(key=lambda x: (x[0][1], x[0][0]))  # Sort by (iy, ix)
    
    print(f"\n📊 Tổng số ô không rỗng: {len(non_empty_cells)}")
    
    for idx, (key, cell) in enumerate(non_empty_cells, 1):
        ix, iy = key
        count = cell.count()
        points = cell.points
        
        print("\n" + "─"*80)
        print(f"🔷 Ô lưới ({ix}, {iy}) - Lần {idx}")
        print("─"*80)
        
        # Thông tin cơ bản
        print(f"\n▶ THÔNG TIN Ô LƯỚI:")
        print(f"   • Vị trí: ({ix}, {iy})")
        print(f"   • Biên X: [{cell.xmin:.4f}, {cell.xmax:.4f}]")
        print(f"   • Biên Y: [{cell.ymin:.4f}, {cell.ymax:.4f}]")
        print(f"   • Số điểm |Mj|: {count}")
        
        # Liệt kê các điểm trong ô
        print(f"\n▶ DANH SÁCH ĐIỂM TRONG Ô:")
        for i, p in enumerate(points):
            print(f"   • Điểm {i+1}: ({p[0]:.4f}, {p[1]:.4f})")
        
        # Bước 1: Kiểm tra mật độ
        print(f"\n▶ BƯỚC 1: KIỂM TRA MẬT ĐỘ")
        print(f"   Công thức: So sánh |Mj| với R")
        print(f"   |Mj| = {count}, R = {R}")
        
        if count == 0:
            print(f"   Kết luận: |Mj| = 0 → ⬜ EMPTY GRID")
            cell.grid_type = 'empty'
            continue
        elif count <= R:
            print(f"   So sánh: {count} ≤ {R}")
            print(f"   Kết luận: 0 < |Mj| ≤ R → 🔵 SPARSE GRID")
            cell.grid_type = 'sparse'
            continue
        else:
            print(f"   So sánh: {count} > {R}")
            print(f"   Kết luận: |Mj| > R → DENSE GRID (cần kiểm tra thêm pj, Dj)")
        
        # Bước 2: Tính Cj (trọng tâm dữ liệu)
        print(f"\n▶ BƯỚC 2: TÍNH TRỌNG TÂM DỮ LIỆU Cj (Equation 4)")
        print(f"   Công thức: Cj = (1/|Mj|) × Σ(xi)")
        
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        Cx = sum_x / count
        Cy = sum_y / count
        
        print(f"\n   Tính Cx:")
        x_values = " + ".join([f"{p[0]:.4f}" for p in points])
        print(f"   Cx = (1/{count}) × ({x_values})")
        print(f"   Cx = (1/{count}) × {sum_x:.4f}")
        print(f"   Cx = {Cx:.4f}")
        
        print(f"\n   Tính Cy:")
        y_values = " + ".join([f"{p[1]:.4f}" for p in points])
        print(f"   Cy = (1/{count}) × ({y_values})")
        print(f"   Cy = (1/{count}) × {sum_y:.4f}")
        print(f"   Cy = {Cy:.4f}")
        
        print(f"\n   ✅ Kết quả: Cj = ({Cx:.4f}, {Cy:.4f})")
        
        # Bước 3: Tính Gj (tâm hình học lưới)
        print(f"\n▶ BƯỚC 3: TÍNH TÂM HÌNH HỌC LƯỚI Gj (Equation 5)")
        print(f"   Công thức: Gj = (Gj_MAX + Gj_MIN) / 2")
        
        Gx = (cell.xmin + cell.xmax) / 2
        Gy = (cell.ymin + cell.ymax) / 2
        
        print(f"\n   Tính Gx:")
        print(f"   Gx = ({cell.xmax:.4f} + {cell.xmin:.4f}) / 2")
        print(f"   Gx = {cell.xmax + cell.xmin:.4f} / 2")
        print(f"   Gx = {Gx:.4f}")
        
        print(f"\n   Tính Gy:")
        print(f"   Gy = ({cell.ymax:.4f} + {cell.ymin:.4f}) / 2")
        print(f"   Gy = {cell.ymax + cell.ymin:.4f} / 2")
        print(f"   Gy = {Gy:.4f}")
        
        print(f"\n   ✅ Kết quả: Gj = ({Gx:.4f}, {Gy:.4f})")
        
        # Bước 4: Tính pj (độ lệch tâm)
        print(f"\n▶ BƯỚC 4: TÍNH ĐỘ LỆCH TÂM pj (Equation 6)")
        print(f"   Công thức: pj = ||Gj - Cj|| / ||Gj_MAX - Gj_MIN||")
        print(f"   Trong đó: ||Gj_MAX - Gj_MIN|| = √[(xmax-xmin)² + (ymax-ymin)²] (đường chéo)")
        
        # Tính khoảng cách |Gj - Cj|
        diff_x = Gx - Cx
        diff_y = Gy - Cy
        numerator = math.sqrt(diff_x**2 + diff_y**2)
        
        print(f"\n   Tính tử số ||Gj - Cj||:")
        print(f"   = √[({Gx:.4f} - {Cx:.4f})² + ({Gy:.4f} - {Cy:.4f})²]")
        print(f"   = √[({diff_x:.4f})² + ({diff_y:.4f})²]")
        print(f"   = √[{diff_x**2:.6f} + {diff_y**2:.6f}]")
        print(f"   = √{diff_x**2 + diff_y**2:.6f}")
        print(f"   = {numerator:.6f}")
        
        # Tính đường chéo
        dx = cell.xmax - cell.xmin
        dy = cell.ymax - cell.ymin
        diagonal = math.sqrt(dx**2 + dy**2)
        
        print(f"\n   Tính mẫu số (đường chéo):")
        print(f"   = √[({cell.xmax:.4f} - {cell.xmin:.4f})² + ({cell.ymax:.4f} - {cell.ymin:.4f})²]")
        print(f"   = √[({dx:.4f})² + ({dy:.4f})²]")
        print(f"   = √[{dx**2:.6f} + {dy**2:.6f}]")
        print(f"   = √{dx**2 + dy**2:.6f}")
        print(f"   = {diagonal:.6f}")
        
        pj = numerator / diagonal if diagonal > 0 else 0
        print(f"\n   Tính pj:")
        print(f"   pj = {numerator:.6f} / {diagonal:.6f}")
        print(f"   pj = {pj:.6f}")
        
        print(f"\n   ✅ Kết quả: pj = {pj:.6f}")
        print(f"   Kiểm tra: pj = {pj:.6f} {'<' if pj < pj_threshold else '>='} {pj_threshold}")
        
        # Bước 5: Tính Dj (độ phân tán)
        print(f"\n▶ BƯỚC 5: TÍNH ĐỘ PHÂN TÁN Dj (Equation 7)")
        print(f"   Công thức: Dj = STPGj / ||Gj_MAX - Gj_MIN||")
        print(f"   Trong đó: STPGj = √[(1/n) × Σ||xi - Cj||²] (độ lệch chuẩn khoảng cách)")
        
        # Tính độ lệch từng điểm đến trọng tâm
        distances_sq = [(p[0] - Cx)**2 + (p[1] - Cy)**2 for p in points]
        
        print(f"\n   Tính khoảng cách từ mỗi điểm đến Cj:")
        for i, (p, d_sq) in enumerate(zip(points, distances_sq)):
            dist = math.sqrt(d_sq)
            print(f"   • Điểm {i+1} ({p[0]:.4f}, {p[1]:.4f}):")
            print(f"     ||xi - Cj||² = ({p[0]:.4f} - {Cx:.4f})² + ({p[1]:.4f} - {Cy:.4f})²")
            print(f"                  = {(p[0] - Cx)**2:.6f} + {(p[1] - Cy)**2:.6f} = {d_sq:.6f}")
        
        mean_dist_sq = sum(distances_sq) / count
        STPGj = math.sqrt(mean_dist_sq)
        
        print(f"\n   Tính STPGj:")
        sum_dist_sq = sum(distances_sq)
        print(f"   Σ||xi - Cj||² = " + " + ".join([f"{d:.4f}" for d in distances_sq]))
        print(f"                 = {sum_dist_sq:.6f}")
        print(f"   (1/n) × Σ||xi - Cj||² = (1/{count}) × {sum_dist_sq:.6f} = {mean_dist_sq:.6f}")
        print(f"   STPGj = √{mean_dist_sq:.6f} = {STPGj:.6f}")
        
        Dj = STPGj / diagonal if diagonal > 0 else 0
        print(f"\n   Tính Dj:")
        print(f"   Dj = STPGj / diagonal")
        print(f"   Dj = {STPGj:.6f} / {diagonal:.6f}")
        print(f"   Dj = {Dj:.6f}")
        
        print(f"\n   ✅ Kết quả: Dj = {Dj:.6f}")
        print(f"   Kiểm tra: Dj = {Dj:.6f} {'<' if Dj < Dj_threshold else '>='} {Dj_threshold}")
        
        # Bước 6: Phân loại cuối cùng
        print(f"\n▶ BƯỚC 6: PHÂN LOẠI")
        print(f"   Điều kiện CORE: pj < {pj_threshold} VÀ Dj < {Dj_threshold}")
        print(f"   Kết quả kiểm tra:")
        print(f"   • pj = {pj:.6f} {'<' if pj < pj_threshold else '>='} {pj_threshold} → {'✓ Thỏa' if pj < pj_threshold else '✗ Không thỏa'}")
        print(f"   • Dj = {Dj:.6f} {'<' if Dj < Dj_threshold else '>='} {Dj_threshold} → {'✓ Thỏa' if Dj < Dj_threshold else '✗ Không thỏa'}")
        
        is_core = pj < pj_threshold and Dj < Dj_threshold
        if is_core:
            cell.grid_type = 'core'
            print(f"\n   ═══════════════════════════════════════════════════════")
            print(f"   🟢 KẾT LUẬN: CORE DENSE GRID (pj < {pj_threshold} VÀ Dj < {Dj_threshold})")
            print(f"   ═══════════════════════════════════════════════════════")
        else:
            cell.grid_type = 'dense'
            reason = []
            if pj >= pj_threshold:
                reason.append(f"pj ≥ {pj_threshold}")
            if Dj >= Dj_threshold:
                reason.append(f"Dj ≥ {Dj_threshold}")
            print(f"\n   ═══════════════════════════════════════════════════════")
            print(f"   🟡 KẾT LUẬN: NON-CORE DENSE GRID ({', '.join(reason)})")
            print(f"   ═══════════════════════════════════════════════════════")
        
        # Bảng tóm tắt cho ô này
        print(f"\n   📋 TÓM TẮT Ô ({ix}, {iy}):")
        print(f"   ┌─────────────┬──────────────────────────────┐")
        print(f"   │ Thông số    │ Giá trị                      │")
        print(f"   ├─────────────┼──────────────────────────────┤")
        print(f"   │ |Mj|        │ {count:<28} │")
        print(f"   │ Cj          │ ({Cx:.4f}, {Cy:.4f}){' '*12} │")
        print(f"   │ Gj          │ ({Gx:.4f}, {Gy:.4f}){' '*12} │")
        print(f"   │ pj          │ {pj:.6f}{' '*21} │")
        print(f"   │ Dj          │ {Dj:.6f}{' '*21} │")
        print(f"   │ Phân loại   │ {cell.grid_type.upper():<28} │")
        print(f"   └─────────────┴──────────────────────────────┘")
    
    # Bảng tổng kết cuối cùng
    print("\n" + "="*80)
    print("BẢNG TỔNG KẾT TẤT CẢ CÁC Ô LƯỚI")
    print("="*80)
    print(f"\n{'Ô lưới':<12} {'|Mj|':<6} {'Cj':<22} {'Gj':<22} {'pj':<10} {'Dj':<10} {'Phân loại':<15}")
    print("-"*105)
    
    for key, cell in sorted(grid.items(), key=lambda x: (x[0][1], x[0][0])):
        count = cell.count()
        if count == 0:
            grid_type = "EMPTY"
            print(f"({cell.ix},{cell.iy}){'':<7} {count:<6} {'-':<22} {'-':<22} {'-':<10} {'-':<10} {grid_type:<15}")
        elif count <= R:
            Cj = cell.centroid()
            Gj = cell.center()
            grid_type = "SPARSE"
            print(f"({cell.ix},{cell.iy}){'':<7} {count:<6} ({Cj[0]:.3f},{Cj[1]:.3f}){'':<10} ({Gj[0]:.3f},{Gj[1]:.3f}){'':<10} {'-':<10} {'-':<10} {grid_type:<15}")
        else:
            Cj = cell.centroid()
            Gj = cell.center()
            pj = cell.compute_pj()
            Dj = cell.compute_Dj()
            grid_type = "CORE" if (pj < pj_threshold and Dj < Dj_threshold) else "NON-CORE"
            print(f"({cell.ix},{cell.iy}){'':<7} {count:<6} ({Cj[0]:.3f},{Cj[1]:.3f}){'':<10} ({Gj[0]:.3f},{Gj[1]:.3f}){'':<10} {pj:.6f}{'  ' if pj < pj_threshold else ' *'} {Dj:.6f}{'  ' if Dj < Dj_threshold else ' *'} {grid_type:<15}")
    
    print("-"*105)
    print(f"Ghi chú: * = không thỏa ngưỡng (pj ≥ {pj_threshold} hoặc Dj ≥ {Dj_threshold})")


# =============================
def plot_classification(points, grid, classified, bounds, M, R):
    """Vẽ kết quả phân loại lưới"""
    xmin, xmax, ymin, ymax = bounds

    fig, ax = plt.subplots(figsize=(12, 10))

    # Màu sắc cho từng loại
    colors = {
        'empty': ('#f0f0f0', 0.3),    # Xám nhạt
        'sparse': ('#87CEEB', 0.4),   # Xanh sky
        'dense': ('#92D050',0.9),    # xanh
        'core': ('#FFFF00',0.9)      # vàng
    }

    # Vẽ các ô lưới theo loại
    labels_drawn = set()

    for grid_type, cells in classified.items():
        color, alpha = colors[grid_type]

        for cell in cells:
            label = None
            if grid_type not in labels_drawn:
                label = f"{grid_type.capitalize()} ({len(cells)})"
                labels_drawn.add(grid_type)

            rect = patches.Rectangle(
                (cell.xmin, cell.ymin),
                cell.xmax - cell.xmin,
                cell.ymax - cell.ymin,
                linewidth=0.5,
                edgecolor='gray',
                facecolor=color,
                alpha=alpha,
                label=label
            )
            ax.add_patch(rect)

    # Vẽ lưới chính MxM (đường đỏ đậm)
    for i in range(M + 1):
        x = xmin + (xmax - xmin) * (i / M)
        y = ymin + (ymax - ymin) * (i / M)

        ax.plot([x, x], [ymin, ymax], 'r-', linewidth=1.0, alpha=0.7)
        ax.plot([xmin, xmax], [y, y], 'r-', linewidth=1.0, alpha=0.7)

    # Vẽ các điểm dữ liệu
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, c='blue', s=30, zorder=10,
               label=f'Data points ({len(points)})')

    # Cài đặt trục: dùng đúng khoảng dữ liệu từng trục
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_aspect('equal')
    ax.set_xlabel('Trục X', fontsize=11)
    ax.set_ylabel('Trục Y', fontsize=11)
    ax.set_title(f' Bước 2: Phân loại lưới: (M={M}x{M}, R={R:.2f})',
                 fontsize=13, fontweight='bold')

    # Legend giải thích màu ô và điểm dữ liệu
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        framealpha=0.9,
    )

    ax.grid(True, alpha=0.2, linestyle='--')

    # Chừa lề phải cho legend để tránh cảnh báo tight_layout
    plt.subplots_adjust(right=0.8)
    plt.show()
# HÀM CHÍNH STEP 2
# =============================
def step2_classify_grids(points, M, R, visualize=True, show_detailed=False):
    """
    Thực hiện Step 2: Phân loại lưới tĩnh MxM

    Args:
        points: Danh sách điểm [(x1,y1), (x2,y2), ...]
        M: Kích thước lưới từ Step 1
        R: Ngưỡng mật độ từ Step 1
        visualize: Có vẽ biểu đồ hay không
        show_detailed: Có in chi tiết từng bước tính toán Cj, Gj, pj, Dj hay không

    Returns:
        dict chứa:
            - grid: toàn bộ lưới
            - classified: các ô đã phân loại
            - bounds: biên dữ liệu
    """

    # 1. Kiểm tra danh sách điểm đầu vào
    is_points_empty = (not points)
    if is_points_empty:
        error_message = (
            "step2_classify_grids: 'points' phải là một danh sách (list) "
            "không rỗng các cặp tọa độ (x, y)."
        )
        raise ValueError(error_message)

    # 2. Kiểm tra tính hợp lệ của M (kích thước lưới)
    is_M_not_integer = not isinstance(M, int)
    is_M_not_positive = M <= 0

    if is_M_not_integer or is_M_not_positive:
        error_message = (
            "step2_classify_grids: 'M' phải là một số nguyên dương "
        )
        raise ValueError(error_message)

    # 3. Kiểm tra tính hợp lệ của R (ngưỡng mật độ)
    is_R_not_positive = R <= 0
    if is_R_not_positive:
        error_message = (
            "step2_classify_grids: 'R' phải là một số thực dương "
            "(lớn hơn 0)."
        )
        raise ValueError(error_message)

    print("STEP 2: GRID CLASSIFICATION")
    print("="*60)
    print(f" Đầu vào: M={M}, R={R:.4f}, Số điểm={len(points)}")

    # 1. Xây dựng lưới MxM
    print(f"\nĐang xây dựng lưới {M}x{M}...")
    grid, bounds = build_grid(points, M)
    print(f"Đã tạo {len(grid)} ô lưới")

    # 2. Phân loại các ô
    print(f"\nĐang phân loại các ô lưới")
    classified = classify_grids(grid, R)

    # 3. In thống kê
    print_statistics(classified, R)

    # 4. In chi tiết từng bước tính toán (nếu được bật)
    if show_detailed:
        # Rebuild grid để in chi tiết (vì classify_grids đã thay đổi grid_type)
        grid_for_detail, _ = build_grid(points, M)
        print_detailed_calculation(grid_for_detail, R)

    # 5. Vẽ biểu đồ
    if visualize:
        print(f"\n Đang vẽ biểu đồ...")
        plot_classification(points, grid, classified, bounds, M, R)

    return {
        'grid': grid,
        'classified': classified,
        'bounds': bounds,
        'M': M,
        'R': R
    }


# =============================
# DEMO SỬ DỤNG
# =============================
if __name__ == "__main__":
    from step1_compute_M_R import step1_compute_original
    from utils import load_data_txt

    data_path = "data.txt"

    # Chạy Step 1
    print(" ĐANG CHẠY STEP 1...")
    step1_result = step1_compute_original(data_path, K=3)

    M = step1_result['M']
    R = step1_result['R']

    # Load dữ liệu
    points = load_data_txt(data_path)

    # Chạy Step 2 với show_detailed=True để in chi tiết từng bước tính toán
    step2_result = step2_classify_grids(points, M, R, visualize=True, show_detailed=True)
    print(" STEP 2 HOÀN THÀNH!")
    print(f"Có {len(step2_result['classified']['core'])} core grids")