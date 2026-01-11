"""Step 3: recursive partitioning of dense grid cells.

Contains functions for splitting dense grid cells recursively and
visualization helpers used by the pipeline.
"""

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from grid_common import (
    diagonal_length,
    compute_pj_for_cell,
    compute_Dj_for_cell,
)

# =============================
# CẬP NHẬT GRIDCELL CHO PHÂN CHIA ĐỆ QUY
# =============================
class RecursiveGridCell:
    """Ô lưới dùng riêng cho Step 3, tự quản lý hình học và thống kê"""

    def __init__(self, ix, iy, xmin, xmax, ymin, ymax, level=1, parent=None, points=None, from_dense_region=False):
        # Thông tin vị trí, hình học
        self.ix = ix
        self.iy = iy
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        # Danh sách điểm trong ô
        self.points = list(points) if points is not None else []

        # Thông tin phân chia đệ quy
        self.level = level  # Cấp độ phân chia (1 là lưới MxM gốc)
        self.parent = parent  # Ô cha
        self.children = []  # Danh sách 4 ô con (nếu có)
        # Cờ: ô này thuộc vùng đã từng được phân loại là 'dense' ở cấp cha
        self.from_dense_region = from_dense_region

        # Mặc định, loại ban đầu là 'unclassified'
        self.grid_type = 'unclassified'

    # ====== Các hàm tiện ích tương tự Step 2 ======
    def add_point(self, p):
        self.points.append(p)

    def count(self):
        return len(self.points)

    def center(self):
        return (
            (self.xmin + self.xmax) / 2.0,
            (self.ymin + self.ymax) / 2.0,
        )

    def centroid(self):
        if not self.points:
            return None
        sx = sum(p[0] for p in self.points)
        sy = sum(p[1] for p in self.points)
        return (sx / len(self.points), sy / len(self.points))

    def _diagonal_length(self):
        return diagonal_length(self.xmin, self.xmax, self.ymin, self.ymax)

    def compute_pj(self):
        """Tính pj giống Step 2 cho ô hiện tại"""
        return compute_pj_for_cell(
            self.points,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
        )

    def compute_Dj(self):
        """Tính Dj giống Step 2 cho ô hiện tại"""
        return compute_Dj_for_cell(
            self.points,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
        )
    #hàm chia đệ quy
    def split_cell(self):
        """Phân chia ô hiện tại thành 4 ô con 2x2"""

        # Các tọa độ chia
        mid_x = (self.xmin + self.xmax) / 2.0
        mid_y = (self.ymin + self.ymax) / 2.0

        # Danh sách 4 ô con (chia theo góc: bottom-left, bottom-right, top-left, top-right)
        new_cells = []

        # Cập nhật tọa độ và tạo ô con (ix, iy là chỉ số trong lưới MxM gốc, không thay đổi)
        # 1. Ô dưới trái (Bottom-Left)
        c1_points = [p for p in self.points if p[0] < mid_x and p[1] < mid_y]
        new_cells.append(RecursiveGridCell(self.ix, self.iy, self.xmin, mid_x, self.ymin, mid_y,
                                           self.level + 1, self, c1_points, from_dense_region=True))

        # 2. Ô dưới phải (Bottom-Right)
        c2_points = [p for p in self.points if p[0] >= mid_x and p[1] < mid_y]
        new_cells.append(RecursiveGridCell(self.ix, self.iy, mid_x, self.xmax, self.ymin, mid_y,
                                           self.level + 1, self, c2_points, from_dense_region=True))

        # 3. Ô trên trái (Top-Left)
        c3_points = [p for p in self.points if p[0] < mid_x and p[1] >= mid_y]
        new_cells.append(RecursiveGridCell(self.ix, self.iy, self.xmin, mid_x, mid_y, self.ymax,
                                           self.level + 1, self, c3_points, from_dense_region=True))

        # 4. Ô trên phải (Top-Right)
        c4_points = [p for p in self.points if p[0] >= mid_x and p[1] >= mid_y]
        new_cells.append(RecursiveGridCell(self.ix, self.iy, mid_x, self.xmax, mid_y, self.ymax,
                                           self.level + 1, self, c4_points, from_dense_region=True))

        self.children = new_cells
        # Đánh dấu ô cha là đã phân chia để không xử lý điểm nữa
        self.grid_type = 'divided'
        return new_cells


# =============================
# PHÂN LOẠI LƯỚI TĨNH (SỬ DỤNG CLASS MỚI)
# =============================
def build_grid_recursive(points, M, bounds):
    """
    Tạo lưới MxM tĩnh ban đầu, sử dụng RecursiveGridCell
    """
    xmin, xmax, ymin, ymax = bounds

    grid = {}

    # Kích thước lưới cố định (MxM)
    for iy in range(M):  # Hàng (Y)
        for ix in range(M):  # Cột (X)
            x0 = xmin + (xmax - xmin) * (ix / M)
            x1 = xmin + (xmax - xmin) * ((ix + 1) / M)
            y0 = ymin + (ymax - ymin) * (iy / M)
            y1 = ymin + (ymax - ymin) * ((iy + 1) / M)

            grid[(ix, iy)] = RecursiveGridCell(ix, iy, x0, x1, y0, y1, level=1)

    # Gán điểm vào các ô (Giống Step 2)
    for p in points:
        x, y = p
        ix = int(min(M - 1, max(0, math.floor((x - xmin) / (xmax - xmin) * M))))
        iy = int(min(M - 1, max(0, math.floor((y - ymin) / (ymax - ymin) * M))))
        grid[(ix, iy)].add_point(p)

    return grid


# =============================
# HÀM PHÂN CHIA ĐỆ QUY
# =============================
def recursive_partitioning(grid, R, max_depth=5):
    """
    Thực hiện phân chia đệ quy cho các ô 'dense'

    Args:
        grid: dict chứa các ô RecursiveGridCell của lưới MxM gốc
        R: Ngưỡng mật độ
        max_depth: Độ sâu phân chia tối đa

    Returns:
        list: Tất cả các ô lưới LÁ (leaf nodes) đã được phân loại cuối cùng
    """
    # Khởi tạo hàng đợi chứa các ô cần kiểm tra/phân chia
    cells_to_process = list(grid.values())
    final_classified_cells = []

    # Lặp lại cho đến khi không còn ô nào cần xử lý
    while cells_to_process:
        current_cell = cells_to_process.pop(0)

        # 1. Phân loại ô hiện tại (dùng hàm đã có từ Step 2)
        # Ta cần một phiên bản classify_grids chỉ nhận 1 ô

        # --- Bắt đầu phân loại (lặp lại logic từ Step 2) ---
        cnt = current_cell.count()

        if cnt == 0:
            current_cell.grid_type = 'empty'
            final_classified_cells.append(current_cell)
            continue

        # Nếu ô KHÔNG thuộc vùng dense trước đó và số điểm không vượt R,
        # ta coi là sparse và dừng luôn (giống Step 2).
        # Ngược lại, với from_dense_region=True, ta muốn tiếp tục kiểm tra
        # core/dense ngay cả khi cnt <= R để giữ lại các ô dense lá.
        if cnt <= R and not getattr(current_cell, 'from_dense_region', False):
            current_cell.grid_type = 'sparse'
            final_classified_cells.append(current_cell)
            continue

        # Các trường hợp còn lại: xử lý như ô dày đặc (ứng viên core/dense)
        pj = current_cell.compute_pj()
        Dj = current_cell.compute_Dj()

        # Ngưỡng core giống Step 2: pj < 0.3 và Dj < 0.7 (nới lỏng để có nhiều core hơn)
        pj_threshold = 0.3
        Dj_threshold = 0.7

        if pj is not None and Dj is not None:
            # Điều kiện dừng theo paper:
            # 1. Đạt max_depth, HOẶC
            # 2. Số điểm < R (không còn đủ dày để chia), HOẶC
            # 3. Ô đã đồng nhất (pj < threshold và Dj < threshold - trở thành core)

            if current_cell.level >= max_depth:
                # Đã đạt độ sâu tối đa - nếu có đủ điểm >= R, coi là core
                if cnt >= R:
                    current_cell.grid_type = 'core'
                else:
                    current_cell.grid_type = 'dense'
                final_classified_cells.append(current_cell)
            elif cnt < R:
                # Không đủ điểm để chia tiếp
                current_cell.grid_type = 'sparse'
                final_classified_cells.append(current_cell)
            elif pj < pj_threshold and Dj < Dj_threshold:
                # Đã đồng nhất -> core
                current_cell.grid_type = 'core'
                final_classified_cells.append(current_cell)
            else:
                # Vẫn còn dense và không đồng nhất -> chia tiếp
                current_cell.grid_type = 'dense'
                new_sub_cells = current_cell.split_cell()
                cells_to_process.extend(new_sub_cells)  # Thêm ô con vào hàng đợi
        else:
            # Không tính được pj, Dj (trường hợp hiếm, coi như sparse leaf)
            current_cell.grid_type = 'sparse'
            final_classified_cells.append(current_cell)

    return final_classified_cells


# =============================
# VẼ BIỂU ĐỒ (CẬP NHẬT CHO CÁC Ô CON)
# =============================
def plot_recursive_classification(points, leaf_cells, bounds, M):
    """Vẽ kết quả phân loại lưới sau phân chia đệ quy"""
    xmin, xmax, ymin, ymax = bounds

    fig, ax = plt.subplots(figsize=(12, 10))

    # Màu sắc cho từng loại (giữ nguyên từ Step 2, nhưng làm dense nổi bật hơn)
    colors = {
        'empty': ('#f0f0f0', 0.3),   # xám nhạt
        'sparse': ('#87CEEB', 0.4),  # xanh dương nhạt
        'dense': ('#00FF00', 0.9),   # xanh lá chói để dễ nhìn
        'core': ('#FFFF00', 1.0),    # vàng kim - nổi bật hơn
        'divided': ('#ffffff', 0.0)  # Bỏ qua ô đã chia
    }

    labels_drawn = set()

    # Vẽ TẤT CẢ các ô LÁ (leaf cells)
    # Để dense dễ thấy, ta vẽ theo thứ tự: empty/sparse trước, sau đó dense, cuối cùng core
    draw_order = ['empty', 'sparse', 'dense', 'core', 'divided']
    for grid_type in draw_order:
        for cell in leaf_cells:
            if cell.grid_type != grid_type:
                continue

            color, alpha = colors[grid_type]

            label = None
            if grid_type not in labels_drawn:
                label = f"{grid_type.capitalize()} ({len([c for c in leaf_cells if c.grid_type == grid_type])})"
                labels_drawn.add(grid_type)

            rect = patches.Rectangle(
                (cell.xmin, cell.ymin),
                cell.xmax - cell.xmin,
                cell.ymax - cell.ymin,
                linewidth=0.5,
                edgecolor='red',  # Đường viền cho ô con
                facecolor=color,
                alpha=alpha,
                label=label
            )
            ax.add_patch(rect)

    # Vẽ các điểm dữ liệu
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, c='blue', s=10, zorder=10,
               label=f'Data points ({len(points)})')

    # Cài đặt trục: dùng đúng bounds từng trục
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_aspect('equal')
    ax.set_xlabel('Trục X', fontsize=11)
    ax.set_ylabel('Trục Y', fontsize=11)
    ax.set_title(f'Step 3: Recursive Grid Partitioning (M={M}x{M} initial)',
                 fontsize=13, fontweight='bold')

    # Legend giải thích màu các loại ô và điểm dữ liệu
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


# =============================
# HÀM CHÍNH STEP 3
# =============================
def step3_handle_dense_grids(points, M, R, bounds, visualize=True, max_depth=5):
    """
    Thực hiện Step 3: Xử lý ô 'dense' bằng phân chia đệ quy
    """
    print("STEP 3: RECURSIVE PARTITIONING (XỬ LÝ Ô DENSE)")
    print(f" Đầu vào: M={M}, R={R:.4f}, Số điểm={len(points)}")

    # 1. Xây dựng lưới MxM ban đầu với RecursiveGridCell
    grid_initial = build_grid_recursive(points, M, bounds)

    # 2. Thực hiện phân chia đệ quy
    print("\n Đang tiến hành phân chia đệ quy các ô 'dense'...")
    print(f"  max_depth = {max_depth}")
    final_classified_cells = recursive_partitioning(grid_initial, R, max_depth=max_depth)
    print(" Phân loại đệ quy hoàn tất.")

    # 3. Thống kê kết quả
    classified_results = {
        'empty': [c for c in final_classified_cells if c.grid_type == 'empty'],
        'sparse': [c for c in final_classified_cells if c.grid_type == 'sparse'],
        'dense': [c for c in final_classified_cells if c.grid_type == 'dense'],
        'core': [c for c in final_classified_cells if c.grid_type == 'core'],
    }

    print("KẾT QUẢ PHÂN LOẠI CUỐI CÙNG (SAU ĐỆ QUY)")
    print(f"📋 Tổng số ô LÁ (leaf cells): {len(final_classified_cells)}")
    print(f"⬜ Empty (leaf):   {len(classified_results['empty']):3d} ô")
    print(f"🔵 Sparse (leaf):  {len(classified_results['sparse']):3d} ô")
    print(f"🟡 Dense (leaf):   {len(classified_results['dense']):3d} ô (không đủ tiêu chí core sau chia)")
    print(f"🟢 Core (leaf):    {len(classified_results['core']):3d} ô")

    # 4. Vẽ biểu đồ
    if visualize:
        print(f"\n Đang vẽ biểu đồ kết quả đệ quy...")
        plot_recursive_classification(points, final_classified_cells, bounds, M)

    return {
        'final_cells': final_classified_cells,
        'classified_results': classified_results,
        'M': M,
        'R': R
    }


# =============================
# DEMO SỬ DỤNG
# =============================
if __name__ == "__main__":
    from step1_compute_M_R import step1_compute_original
    from utils import load_data_txt
    from step2_grid_classification import build_grid, classify_grids, plot_classification

    data_path = "data.txt"

    # --- CHẠY STEP 1: TÌM M & R ---
    try:
        print(" ĐANG CHẠY STEP 1...")
        step1_result = step1_compute_original(data_path, K=3, max_M=200)
        M = step1_result['M']
        R = step1_result['R']
        points = load_data_txt(data_path)
    except Exception as e:
        print(f"Lỗi khi chạy Step 1: {e}")
        exit()

    print("\n--- HIỂN THỊ KẾT QUẢ STEP 2 (Phân loại tĩnh) ---")
    grid, bounds = build_grid(points, M)
    classified = classify_grids(grid, R)
    plot_classification(points, grid, classified, bounds, M, R)
    # ==================================

    # --- CHẠY STEP 3: PHÂN CHIA ĐỆ QUY ---
    step3_result = step3_handle_dense_grids(points, M, R, bounds, visualize=True)
    print(" STEP 3 HOÀN THÀNH!")
    print(f" Có {len(step3_result['classified_results']['core'])} core grids cuối cùng.")
