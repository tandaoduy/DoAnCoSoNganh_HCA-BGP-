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
def classify_grids(grid, R, pj_threshold=0.05, Dj_threshold=0.3):
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
    print(f"🟢 Core grids:     {len(classified['core']):3d} ô (count > R, pj<0.05, Dj<0.3)")

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
    ax.set_title(f'Step 2: Grid Classification (M={M}x{M}, R={R:.4f})',
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
def step2_classify_grids(points, M, R, visualize=True):
    """
    Thực hiện Step 2: Phân loại lưới tĩnh MxM

    Args:
        points: Danh sách điểm [(x1,y1), (x2,y2), ...]
        M: Kích thước lưới từ Step 1
        R: Ngưỡng mật độ từ Step 1
        visualize: Có vẽ biểu đồ hay không

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

    # 4. Vẽ biểu đồ
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

    # Chạy Step 2
    step2_result = step2_classify_grids(points, M, R, visualize=True)
    print(" STEP 2 HOÀN THÀNH!")
    print(f"Có {len(step2_result['classified']['core'])} core grids")