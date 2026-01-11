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

        # Ngưỡng core: pj < pj_threshold và Dj < Dj_threshold
        # Tăng ngưỡng để nhiều ô hơn đạt tiêu chuẩn core
        pj_threshold = 0.5  # tăng từ 0.1 lên 0.5
        Dj_threshold = 0.8  # tăng từ 0.5 lên 0.8

        if pj is not None and Dj is not None:
            # Điều kiện dừng theo paper:
            # 1. Đạt max_depth, HOẶC
            # 2. Số điểm < R (không còn đủ dày để chia), HOẶC
            # 3. Ô đã đồng nhất (pj < threshold và Dj < threshold - trở thành core)

            if current_cell.level >= max_depth:
                # Đã đạt độ sâu tối đa, giữ nguyên phân loại hiện tại (dense leaf)
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
# IN CHI TIẾT QUÁ TRÌNH CHIA ĐỆ QUY
# =============================
def print_recursive_partitioning_detail(grid, R, max_depth=5, pj_threshold=0.5, Dj_threshold=0.8):
    """
    In chi tiết từng bước của quá trình chia lưới đệ quy.
    Hiển thị cách tính Cj, Gj, pj, Dj và quyết định chia/dừng.
    """
    print("\n" + "="*90)
    print("CHI TIẾT QUÁ TRÌNH CHIA LƯỚI ĐỆ QUY (STEP 3)")
    print("="*90)
    print(f"\n📌 Ngưỡng mật độ R = {R}")
    print(f"📌 Ngưỡng pj < {pj_threshold} (độ lệch tâm)")
    print(f"📌 Ngưỡng Dj < {Dj_threshold} (độ phân tán)")
    print(f"📌 Độ sâu tối đa max_depth = {max_depth}")
    print(f"\n📖 NGUYÊN LÝ: Ô Dense sẽ được chia đôi mỗi chiều → tạo 2^T = 4 ô con (T=2 chiều)")
    print("   Tiếp tục chia cho đến khi: (1) đạt max_depth, (2) count < R, hoặc (3) ô trở thành Core")
    
    # Khởi tạo hàng đợi
    cells_to_process = list(grid.values())
    final_cells = []
    step_counter = 0
    
    print(f"\n📊 Tổng số ô ban đầu: {len(cells_to_process)}")
    
    while cells_to_process:
        current_cell = cells_to_process.pop(0)
        step_counter += 1
        
        cnt = current_cell.count()
        level = current_cell.level
        
        # Chỉ in chi tiết cho các ô có điểm và thuộc vùng dense hoặc có count > R
        if cnt == 0:
            current_cell.grid_type = 'empty'
            final_cells.append(current_cell)
            continue
        
        if cnt <= R and not getattr(current_cell, 'from_dense_region', False):
            current_cell.grid_type = 'sparse'
            final_cells.append(current_cell)
            continue
        
        # In chi tiết cho ô Dense
        print("\n" + "─"*90)
        print(f"🔷 BƯỚC {step_counter}: Ô tại Level {level}")
        print("─"*90)
        
        # Thông tin ô lưới
        print(f"\n▶ THÔNG TIN Ô LƯỚI:")
        print(f"   • Vị trí gốc (ix, iy): ({current_cell.ix}, {current_cell.iy})")
        print(f"   • Level (cấp độ chia): {level}")
        print(f"   • Biên X: [{current_cell.xmin:.4f}, {current_cell.xmax:.4f}]")
        print(f"   • Biên Y: [{current_cell.ymin:.4f}, {current_cell.ymax:.4f}]")
        print(f"   • Kích thước: ΔX = {current_cell.xmax - current_cell.xmin:.4f}, ΔY = {current_cell.ymax - current_cell.ymin:.4f}")
        print(f"   • Số điểm |Mj|: {cnt}")
        print(f"   • Thuộc vùng Dense trước: {'Có' if current_cell.from_dense_region else 'Không'}")
        
        # Liệt kê các điểm
        print(f"\n▶ DANH SÁCH ĐIỂM TRONG Ô:")
        for i, p in enumerate(current_cell.points):
            print(f"   • Điểm {i+1}: ({p[0]:.4f}, {p[1]:.4f})")
        
        # Tính toán Cj, Gj, pj, Dj
        pj = current_cell.compute_pj()
        Dj = current_cell.compute_Dj()
        Cj = current_cell.centroid()
        Gj = current_cell.center()
        
        # Tính chi tiết
        print(f"\n▶ TÍNH TRỌNG TÂM DỮ LIỆU Cj:")
        if Cj:
            sum_x = sum(p[0] for p in current_cell.points)
            sum_y = sum(p[1] for p in current_cell.points)
            print(f"   Cx = (1/{cnt}) × Σ(xi) = (1/{cnt}) × {sum_x:.4f} = {Cj[0]:.4f}")
            print(f"   Cy = (1/{cnt}) × Σ(yi) = (1/{cnt}) × {sum_y:.4f} = {Cj[1]:.4f}")
            print(f"   ✅ Cj = ({Cj[0]:.4f}, {Cj[1]:.4f})")
        
        print(f"\n▶ TÍNH TÂM HÌNH HỌC LƯỚI Gj:")
        print(f"   Gx = ({current_cell.xmax:.4f} + {current_cell.xmin:.4f}) / 2 = {Gj[0]:.4f}")
        print(f"   Gy = ({current_cell.ymax:.4f} + {current_cell.ymin:.4f}) / 2 = {Gj[1]:.4f}")
        print(f"   ✅ Gj = ({Gj[0]:.4f}, {Gj[1]:.4f})")
        
        # Tính pj
        print(f"\n▶ TÍNH ĐỘ LỆCH TÂM pj:")
        if Cj and pj is not None:
            diff_x = Gj[0] - Cj[0]
            diff_y = Gj[1] - Cj[1]
            numerator = math.sqrt(diff_x**2 + diff_y**2)
            dx = current_cell.xmax - current_cell.xmin
            dy = current_cell.ymax - current_cell.ymin
            diagonal = math.sqrt(dx**2 + dy**2)
            print(f"   ||Gj - Cj|| = √[({Gj[0]:.4f} - {Cj[0]:.4f})² + ({Gj[1]:.4f} - {Cj[1]:.4f})²]")
            print(f"              = √[{diff_x**2:.6f} + {diff_y**2:.6f}] = {numerator:.6f}")
            print(f"   Đường chéo = √[{dx:.4f}² + {dy:.4f}²] = {diagonal:.6f}")
            print(f"   pj = {numerator:.6f} / {diagonal:.6f} = {pj:.6f}")
            print(f"   ✅ pj = {pj:.6f} {'<' if pj < pj_threshold else '>='} {pj_threshold}")
        
        # Tính Dj
        print(f"\n▶ TÍNH ĐỘ PHÂN TÁN Dj:")
        if Cj and Dj is not None:
            distances_sq = [(p[0] - Cj[0])**2 + (p[1] - Cj[1])**2 for p in current_cell.points]
            mean_dist_sq = sum(distances_sq) / cnt
            STPGj = math.sqrt(mean_dist_sq)
            dx = current_cell.xmax - current_cell.xmin
            dy = current_cell.ymax - current_cell.ymin
            diagonal = math.sqrt(dx**2 + dy**2)
            print(f"   Σ||xi - Cj||² = " + " + ".join([f"{d:.4f}" for d in distances_sq[:5]]) + ("..." if cnt > 5 else ""))
            print(f"                 = {sum(distances_sq):.6f}")
            print(f"   STPGj = √[(1/{cnt}) × {sum(distances_sq):.6f}] = {STPGj:.6f}")
            print(f"   Dj = {STPGj:.6f} / {diagonal:.6f} = {Dj:.6f}")
            print(f"   ✅ Dj = {Dj:.6f} {'<' if Dj < Dj_threshold else '>='} {Dj_threshold}")
        
        # Quyết định
        print(f"\n▶ KIỂM TRA ĐIỀU KIỆN VÀ QUYẾT ĐỊNH:")
        print(f"   Điều kiện 1: Level ({level}) >= max_depth ({max_depth})? → {'CÓ' if level >= max_depth else 'KHÔNG'}")
        print(f"   Điều kiện 2: count ({cnt}) < R ({R})? → {'CÓ' if cnt < R else 'KHÔNG'}")
        print(f"   Điều kiện 3: pj ({pj:.4f}) < {pj_threshold} VÀ Dj ({Dj:.4f}) < {Dj_threshold}? → {'CÓ (CORE)' if (pj < pj_threshold and Dj < Dj_threshold) else 'KHÔNG'}")
        
        if level >= max_depth:
            current_cell.grid_type = 'dense'
            final_cells.append(current_cell)
            print(f"\n   ═══════════════════════════════════════════════════════════════")
            print(f"   🟡 KẾT LUẬN: DENSE (đạt max_depth={max_depth}) → DỪNG CHIA")
            print(f"   ═══════════════════════════════════════════════════════════════")
        elif cnt < R:
            current_cell.grid_type = 'sparse'
            final_cells.append(current_cell)
            print(f"\n   ═══════════════════════════════════════════════════════════════")
            print(f"   🔵 KẾT LUẬN: SPARSE (count < R) → DỪNG CHIA")
            print(f"   ═══════════════════════════════════════════════════════════════")
        elif pj < pj_threshold and Dj < Dj_threshold:
            current_cell.grid_type = 'core'
            final_cells.append(current_cell)
            print(f"\n   ═══════════════════════════════════════════════════════════════")
            print(f"   🟢 KẾT LUẬN: CORE (pj < {pj_threshold} VÀ Dj < {Dj_threshold}) → DỪNG CHIA")
            print(f"   ═══════════════════════════════════════════════════════════════")
        else:
            # Chia lưới
            print(f"\n   ═══════════════════════════════════════════════════════════════")
            print(f"   🔶 KẾT LUẬN: DENSE + KHÔNG ĐỒNG NHẤT → TIẾP TỤC CHIA")
            print(f"   ═══════════════════════════════════════════════════════════════")
            
            # Hiển thị chi tiết cách chia
            mid_x = (current_cell.xmin + current_cell.xmax) / 2.0
            mid_y = (current_cell.ymin + current_cell.ymax) / 2.0
            
            print(f"\n▶ CHIA Ô THÀNH 4 Ô CON (2x2):")
            print(f"   Tâm chia: mid_X = ({current_cell.xmin:.4f} + {current_cell.xmax:.4f})/2 = {mid_x:.4f}")
            print(f"             mid_Y = ({current_cell.ymin:.4f} + {current_cell.ymax:.4f})/2 = {mid_y:.4f}")
            
            # Tính số điểm mỗi ô con
            c1_points = [p for p in current_cell.points if p[0] < mid_x and p[1] < mid_y]
            c2_points = [p for p in current_cell.points if p[0] >= mid_x and p[1] < mid_y]
            c3_points = [p for p in current_cell.points if p[0] < mid_x and p[1] >= mid_y]
            c4_points = [p for p in current_cell.points if p[0] >= mid_x and p[1] >= mid_y]
            
            print(f"\n   ┌─────────────────────────────────────────────────────────────────────────────┐")
            print(f"   │ Ô con        │ X range                    │ Y range                    │ Điểm │")
            print(f"   ├─────────────────────────────────────────────────────────────────────────────┤")
            print(f"   │ Bottom-Left  │ [{current_cell.xmin:.4f}, {mid_x:.4f})  │ [{current_cell.ymin:.4f}, {mid_y:.4f})  │ {len(c1_points):4} │")
            print(f"   │ Bottom-Right │ [{mid_x:.4f}, {current_cell.xmax:.4f})  │ [{current_cell.ymin:.4f}, {mid_y:.4f})  │ {len(c2_points):4} │")
            print(f"   │ Top-Left     │ [{current_cell.xmin:.4f}, {mid_x:.4f})  │ [{mid_y:.4f}, {current_cell.ymax:.4f})  │ {len(c3_points):4} │")
            print(f"   │ Top-Right    │ [{mid_x:.4f}, {current_cell.xmax:.4f})  │ [{mid_y:.4f}, {current_cell.ymax:.4f})  │ {len(c4_points):4} │")
            print(f"   └─────────────────────────────────────────────────────────────────────────────┘")
            print(f"   → Tổng: {len(c1_points)} + {len(c2_points)} + {len(c3_points)} + {len(c4_points)} = {cnt} điểm")
            
            # Thực hiện chia
            current_cell.grid_type = 'divided'
            new_cells = current_cell.split_cell()
            cells_to_process.extend(new_cells)
            print(f"   → Thêm 4 ô con vào hàng đợi xử lý (Level {level+1})")
        
        # Bảng tóm tắt
        print(f"\n   📋 TÓM TẮT:")
        print(f"   ┌───────────────┬────────────────────────────────┐")
        print(f"   │ Thông số      │ Giá trị                        │")
        print(f"   ├───────────────┼────────────────────────────────┤")
        print(f"   │ Level         │ {level:<30} │")
        print(f"   │ |Mj|          │ {cnt:<30} │")
        if Cj:
            print(f"   │ Cj            │ ({Cj[0]:.4f}, {Cj[1]:.4f}){' '*14} │")
        print(f"   │ Gj            │ ({Gj[0]:.4f}, {Gj[1]:.4f}){' '*14} │")
        print(f"   │ pj            │ {pj:.6f}{' '*23} │")
        print(f"   │ Dj            │ {Dj:.6f}{' '*23} │")
        print(f"   │ Phân loại     │ {current_cell.grid_type.upper():<30} │")
        print(f"   └───────────────┴────────────────────────────────┘")
    
    # Bảng tổng kết
    print("\n" + "="*90)
    print("BẢNG TỔNG KẾT CÁC Ô LÁ (LEAF CELLS) SAU CHIA ĐỆ QUY")
    print("="*90)
    
    core_cells = [c for c in final_cells if c.grid_type == 'core']
    dense_cells = [c for c in final_cells if c.grid_type == 'dense']
    sparse_cells = [c for c in final_cells if c.grid_type == 'sparse']
    empty_cells = [c for c in final_cells if c.grid_type == 'empty']
    
    print(f"\n📊 THỐNG KÊ:")
    print(f"   🟢 Core cells:   {len(core_cells)} ô")
    print(f"   🟡 Dense cells:  {len(dense_cells)} ô")
    print(f"   🔵 Sparse cells: {len(sparse_cells)} ô")
    print(f"   ⬜ Empty cells:  {len(empty_cells)} ô")
    print(f"   ─────────────────────")
    print(f"   📋 Tổng leaf cells: {len(final_cells)} ô")
    
    if core_cells:
        print(f"\n📍 CHI TIẾT CÁC CORE CELLS:")
        print(f"{'#':<4} {'Level':<6} {'X range':<25} {'Y range':<25} {'Điểm':<6} {'pj':<10} {'Dj':<10}")
        print("-"*90)
        for i, c in enumerate(core_cells, 1):
            pj = c.compute_pj()
            Dj = c.compute_Dj()
            x_range = f"[{c.xmin:.3f}, {c.xmax:.3f})"
            y_range = f"[{c.ymin:.3f}, {c.ymax:.3f})"
            print(f"{i:<4} {c.level:<6} {x_range:<25} {y_range:<25} {c.count():<6} {pj:.6f}   {Dj:.6f}")
    
    return final_cells


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
    ax.set_title(f'Bước 3: Chia đệ quy (M={M}x{M})',
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
def step3_handle_dense_grids(points, M, R, bounds, visualize=True, max_depth=5, show_detailed=False):
    """
    Thực hiện Step 3: Xử lý ô 'dense' bằng phân chia đệ quy
    
    Args:
        points: Danh sách điểm [(x1,y1), (x2,y2), ...]
        M: Kích thước lưới từ Step 1
        R: Ngưỡng mật độ từ Step 1
        bounds: Biên dữ liệu (xmin, xmax, ymin, ymax)
        visualize: Có vẽ biểu đồ hay không
        max_depth: Độ sâu tối đa của phân chia đệ quy
        show_detailed: Có in chi tiết từng bước chia lưới đệ quy hay không
    """
    print("Bước 3: Chia đệ quy (XỬ LÝ Ô DENSE)")
    print(f" Đầu vào: M={M}, R={R:.4f}, Số điểm={len(points)}")

    # 1. Xây dựng lưới MxM ban đầu với RecursiveGridCell
    grid_initial = build_grid_recursive(points, M, bounds)

    # 2. In chi tiết quá trình chia đệ quy (nếu được bật)
    if show_detailed:
        # Tạo lại grid để in chi tiết (vì recursive_partitioning sẽ thay đổi grid)
        grid_for_detail = build_grid_recursive(points, M, bounds)
        print_recursive_partitioning_detail(grid_for_detail, R, max_depth=max_depth)
    
    # 3. Thực hiện phân chia đệ quy
    print("\n Đang tiến hành phân chia đệ quy các ô 'dense'...")
    print(f"  max_depth = {max_depth}")
    final_classified_cells = recursive_partitioning(grid_initial, R, max_depth=max_depth)
    print(" Phân loại đệ quy hoàn tất.")

    # 4. Thống kê kết quả
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

    # 5. Vẽ biểu đồ
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

    # --- CHẠY STEP 3: PHÂN CHIA ĐỆ QUY (với show_detailed=True để in chi tiết) ---
    step3_result = step3_handle_dense_grids(points, M, R, bounds, visualize=True, show_detailed=True)
    print(" STEP 3 HOÀN THÀNH!")
    print(f" Có {len(step3_result['classified_results']['core'])} core grids cuối cùng.")
