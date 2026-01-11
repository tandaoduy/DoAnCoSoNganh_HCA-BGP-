"""Step 1: compute M and R for HCA-BGP.

This module contains legacy functions used to compute grid size M and radius R
from input point data. The functions are intentionally kept compatible with
the original pipeline and called by other steps.
"""

import statistics
import matplotlib.pyplot as plt
from utils import load_data_txt


# ============================================
# HÀM VẼ LƯỚI STEP 1 (BẢN CŨ - LEGACY)
# ============================================
def plot_step1_result_legacy(points, M, show_labels=False):
    # Tách danh sách điểm thành hai danh sách riêng:
    #   - x_coords: tất cả hoành độ (x)
    #   - y_coords: tất cả tung độ (y)
    x_coords, y_coords = zip(*points)

    # Xác định miền bao trùm dữ liệu theo từng trục
    min_x_value = min(x_coords)
    max_x_value = max(x_coords)
    min_y_value = min(y_coords)
    max_y_value = max(y_coords)

    # Tạo figure và axes để vẽ với kích thước 8x8 inch
    figure, axis = plt.subplots(figsize=(8, 8))

    # Vẽ tất cả các điểm dữ liệu lên đồ thị
    #   - s=80: kích thước marker
    #   - color='blue': màu xanh dương
    #   - marker='.': dạng chấm
    axis.scatter(
        x_coords,
        y_coords,
        s=80,
        color='blue',
        marker='.'
    )

    # VẼ LƯỚI VUÔNG M x M
    x_grid_positions = []  # lưu lại các giá trị X của đường lưới dọc
    y_grid_positions = []  # lưu lại các giá trị Y của đường lưới ngang

    # i chạy từ 0 đến M (bao gồm M)
    #   - i = 0   → biên trái / biên dưới
    #   - i = M   → biên phải / biên trên
    #   - 0 < i < M → các đường lưới nằm bên trong
    for grid_index in range(M + 1):
        # Tỉ lệ vị trí từ 0.0 đến 1.0
        position_ratio = grid_index / M
        # Tọa độ đường lưới theo trục X và trục Y
        x_position = min_x_value + (max_x_value - min_x_value) * position_ratio
        y_position = min_y_value + (max_y_value - min_y_value) * position_ratio
        # Lưu lại để tham khảo
        x_grid_positions.append(x_position)
        y_grid_positions.append(y_position)
        # Vẽ đường thẳng đứng tại x = x_position
        axis.axvline(
            x_position,
            color='red',
            linewidth=1.0,
            alpha=0.7
        )
        # Vẽ đường thẳng ngang tại y = y_position
        axis.axhline(
            y_position,
            color='red',
            linewidth=1.0,
            alpha=0.7
        )

    # CÀI ĐẶT TRỤC VÀ HÌNH DÁNG
    # Giới hạn trục X và Y đúng bằng miền dữ liệu
    axis.set_xlim(min_x_value, max_x_value)
    axis.set_ylim(min_y_value, max_y_value)
    # Đảm bảo tỉ lệ hai trục là như nhau (ô vuông không bị méo)
    axis.set_aspect('equal')
    # Tắt lưới mặc định rồi bật lại lưới nền mờ để dễ nhìn
    axis.grid(False)
    axis.grid(
        True,
        alpha=0.2,
        linestyle='--'
    )
    # Bỏ dạng hiển thị "offset" trên trục X (vd: 1e6)
    # để hiển thị số thật đầy đủ
    axis.ticklabel_format(
        style='plain',
        axis='x'
    )
    axis.set_title(f"Step1: Vẽ lưới  M={M}")
    plt.show()



# HÀM TÍNH Mdg(M) (BẢN CŨ - LEGACY)
def compute_Mdg_legacy(points, M):
    """
    Ý tưởng:
        - Chia không gian dữ liệu thành lưới M x M.
        - Đếm số điểm rơi vào từng ô.
        - Lấy tất cả ô có ít nhất 1 điểm.
        - Trả về median (trung vị) của các số đếm đó.
    Returns:
        Mdg(M) = median số điểm trong các ô khác rỗng.
        Nếu không có ô nào có điểm → trả về 0.
    """

    # Tách các tọa độ X, Y riêng để dễ xử lý
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    # Xác định miền bao trùm dữ liệu
    min_x_value = min(x_coords)
    max_x_value = max(x_coords)
    min_y_value = min(y_coords)
    max_y_value = max(y_coords)

    # Tránh trường hợp tất cả điểm trùng nhau trên 1 trục (chia cho 0)
    if max_x_value == min_x_value:
        max_x_value = max_x_value + 1

    if max_y_value == min_y_value:
        max_y_value = max_y_value + 1

    # Tạo ma trận đếm số điểm trong từng ô lưới:
    #   - counts[row_index][column_index]
    #   - Kích thước: M hàng x M cột
    cell_point_counts = []
    for row_index in range(M):
        row_counts = []
        for column_index in range(M):
            row_counts.append(0)
        cell_point_counts.append(row_counts)

    # Cho từng điểm (x, y) vào đúng ô trong lưới
    for point in points:
        point_x = point[0]
        point_y = point[1]

        # Vị trí tương đối theo trục X và Y (0.0 → 1.0)
        relative_x = (point_x - min_x_value) / (max_x_value - min_x_value)
        relative_y = (point_y - min_y_value) / (max_y_value - min_y_value)

        # Chuyển sang chỉ số cột / hàng trong lưới
        cell_column_index = int(relative_x * M)
        cell_row_index = int(relative_y * M)

        # Nếu đúng biên phải / trên thì đưa về ô cuối cùng (M - 1)
        if cell_column_index == M:
            cell_column_index = M - 1

        if cell_row_index == M:
            cell_row_index = M - 1

        # Tăng bộ đếm số điểm cho ô tương ứng
        cell_point_counts[cell_row_index][cell_column_index] += 1

    # Thu thập tất cả các ô có ít nhất 1 điểm
    non_empty_cell_counts = []
    for row_index in range(M):
        for column_index in range(M):
            current_cell_count = cell_point_counts[row_index][column_index]
            if current_cell_count > 0:
                non_empty_cell_counts.append(current_cell_count)

    # Nếu có ít nhất một ô chứa điểm → trả về median.
    # Nếu không có ô nào chứa điểm → trả về 0.
    if non_empty_cell_counts:
        median_value = statistics.median(non_empty_cell_counts)
        return median_value
    else:
        return 0


# STEP 1 – ĐÚNG THEO BÀI GỐC (BẢN CŨ - LEGACY)
def step1_compute_original_legacy(path, K=3, max_M=500):
    """
    Thực hiện Step 1 của thuật toán HCA-BGP đúng theo bài gốc:

    - Đọc dữ liệu điểm từ file.
    - Tính |Umin| = n / (10 * K).
    - Quét M theo các giá trị: K, 2K, 3K, ..., <= max_M.
    - Tại mỗi M, tính Mdg(M).
    - Chọn M* là giá trị M có Mdg(M) nhỏ nhất (nếu có nhiều, lấy M đầu tiên).
    - Tính R_gốc = Mdg(M*) / 2.
    - Áp dụng clamp: R = max(1.0, R_gốc) để dùng cho các bước sau.
    """
    # Đọc dữ liệu điểm từ file
    points = load_data_txt(path)
    number_of_points = len(points)

    print("===== HCA-BGP STEP 1 =====")
    print(f"Số điểm: {number_of_points} | K mong muốn: {K}")

    # === Tính |Umin| theo bài báo gốc ===
    Umin = number_of_points / (10 * K)
    print(f"|Umin| (theo bài gốc) = {Umin:.3f}")
    # === Khởi tạo  M  = K===
    current_grid_size_M = K

    # best_M: giá trị M tốt nhất tìm được cho tới thời điểm hiện tại
    # best_Mdg: giá trị Mdg(M) nhỏ nhất tương ứng với best_M
    best_grid_size_M = None
    best_Mdg_value = None

    # =========================================
    # VÒNG LẶP QUÉT M THEO BỘI SỐ CỦA K
    # =========================================
    # Duyệt M = K, 2K, 3K, ... <= max_M
    # Tại mỗi M, ta tính Mdg(M).
    # - Nếu đây là lần đầu tiên (best_Mdg_value is None) → gán luôn.
    # - Hoặc nếu Mdg(M) < best_Mdg_value hiện tại → cập nhật best_M, best_Mdg.
    # - Nếu Mdg(M) == best_Mdg_value hiện tại → KHÔNG cập nhật
    #   (tức là giữ lại M đầu tiên đạt giá trị nhỏ nhất đó).
    while current_grid_size_M <= max_M:
        current_Mdg_value = compute_Mdg_legacy(points, current_grid_size_M)
        print(
            f"  M={current_grid_size_M:4d}  \t Mdg={current_Mdg_value:.3f}"
        )

        # Cập nhật nghiệm tốt nhất nếu:
        #  - Chưa có nghiệm nào (best_Mdg_value is None), hoặc
        #  - Vừa tìm được M với Mdg nhỏ hơn rõ rệt.
        if best_Mdg_value is None or current_Mdg_value < best_Mdg_value:
            best_grid_size_M = current_grid_size_M
            best_Mdg_value = current_Mdg_value

        # Tăng M theo đúng quy tắc gốc: cộng thêm K
        current_grid_size_M = current_grid_size_M + K

    # Nếu vì lý do nào đó không tính được Mdg (trường hợp hiếm),
    # ta fallback về M = 6 để tránh lỗi chia cho 0 hoặc danh sách rỗng.
    if best_grid_size_M is None:
        print("Không tính được Mdg hợp lệ, dùng mặc định M = 6.")
        best_grid_size_M = 6
        best_Mdg_value = compute_Mdg_legacy(points, best_grid_size_M)

    # === Tính R theo bài gốc ===
    R_original = best_Mdg_value / 2.0

    # Để tránh trường hợp R quá nhỏ (ví dụ < 1) làm mất hẳn phân lớp sparse,
    # ta ép R tối thiểu là 1.0 khi sử dụng cho các bước sau.
    R_for_next_steps = max(1.0, R_original)

    print("\n===== KẾT QUẢ STEP 1 (BẢN GỐC) =====")
    print(f"Số điểm: {number_of_points}")
    print(f"K mong muốn: {K}")
    print(f"|Umin| = {Umin:.3f}")
    print(f"Chọn M* = {best_grid_size_M}")
    print(f"Mdg(M*) = {best_Mdg_value:.3f}")
    print(f"R_goc (Mdg/2) = {R_original:.3f}")
    print(f"R_dung (sau khi clamp >= 1.0) = {R_for_next_steps:.3f}")
    print("========================================")

    # Vẽ lưới với M tối ưu
    plot_step1_result_legacy(points, best_grid_size_M, show_labels=False)

    # Trả về kết quả để các step sau sử dụng
    return {
        "M": best_grid_size_M,
        "Mdg": best_Mdg_value,
        "R": R_for_next_steps,
        "Umin": Umin
    }


# Wrapper giữ nguyên tên hàm cũ cho các step sau
# Mặc định quay lại dùng bản gốc (legacy) để đảm bảo kết quả phân cụm
# giống bài báo và các file silhouette_results ban đầu.
def step1_compute_original(path, K=3, max_M=500):
    return step1_compute_original_legacy(path, K=K, max_M=max_M)


if __name__ == "__main__":
    step1_compute_original_legacy("data.txt", K=3, max_M=200)