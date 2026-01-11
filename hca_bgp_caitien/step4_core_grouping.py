"""Step 4: group core grids into core-clusters.

Provides functions to aggregate adjacent core grid cells into core-clusters
and plotting helpers used by the pipeline.
"""

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ------------------------------------------------------
# Tính khoảng cách giữa 2 grid theo công thức (9), (10)
# ------------------------------------------------------
def axis_distance(g1, g2, axis):
    """Khoảng cách 1 chiều giữa 2 lưới a,b theo Definition 9.

       distance_i(a,b) = min(|maxA-maxB|, |minA-minB|) / min(lenA, lenB)
       với lenA = maxA-minA, lenB = maxB-minB.
    """
    a_min, a_max = g1["min_bin"][axis], g1["max_bin"][axis]
    b_min, b_max = g2["min_bin"][axis], g2["max_bin"][axis]

    len_a = abs(a_max - a_min)
    len_b = abs(b_max - b_min)
    denom = min(len_a, len_b)
    if denom == 0:
        return 0.0

    num = min(abs(a_max - b_max), abs(a_min - b_min))
    return num / denom


def grid_distance(g1, g2, dim):
    """Khoảng cách N chiều giữa 2 lưới a,b theo Definition 10.

       distance(a,b)_N = sum_i distance_i(a,b)
    """
    return sum(axis_distance(g1, g2, d) for d in range(dim))


# ------------------------------------------------------
# Kiểm tra adjacency theo bài báo: ô kề nhau theo (ix, iy)
# ------------------------------------------------------
def are_adjacent(g1, g2):
    """Hai grid được coi là kề nhau nếu chỉ số ix, iy chênh nhau không quá 1."""
    return abs(g1["ix"] - g2["ix"]) <= 1 and abs(g1["iy"] - g2["iy"]) <= 1


# ------------------------------------------------------
# Gom các core-grid thành các core-cluster (Step 4)
# ------------------------------------------------------
def build_core_clusters(grid_list, dim):
    core_grids = [g for g in grid_list if g["is_core"]]
    N = len(core_grids)

    # adjacency matrix: 1 nếu hai core-grid được xem là liền kề
    # - Nếu có chỉ số (ix, iy) -> dùng are_adjacent theo bài báo (ô kề nhau)
    # - Nếu KHÔNG có ix/iy (ví dụ grid từ lưới đệ quy) -> fallback dùng grid_distance < 1.0
    adj = [[0] * N for _ in range(N)]

    for i in range(N):
        for j in range(i + 1, N):
            g1 = core_grids[i]
            g2 = core_grids[j]
            if "ix" in g1 and "iy" in g1 and "ix" in g2 and "iy" in g2:
                # lưới tĩnh Step 2: adjacency theo ô kề (ix, iy)
                if are_adjacent(g1, g2):
                    adj[i][j] = adj[j][i] = 1
            else:
                # lưới đệ quy hoặc format khác: dùng khoảng cách chuẩn hoá < 1
                d = grid_distance(g1, g2, dim)
                if d < 1.0:
                    adj[i][j] = adj[j][i] = 1

    # BFS grouping
    visited = [False] * N
    clusters = []
    for i in range(N):
        if not visited[i]:
            q = deque([i])
            visited[i] = True
            comp = [i]

            while q:
                u = q.popleft()
                for v in range(N):
                    if adj[u][v] == 1 and not visited[v]:
                        visited[v] = True
                        q.append(v)
                        comp.append(v)

            clusters.append([core_grids[idx] for idx in comp])

    return clusters


# ------------------------------------------------------
# Tính centroid mỗi core-grid (step 6)
# ------------------------------------------------------
def compute_coregrid_centroid(grid):
    pts = np.array(grid["points"])
    if len(pts) == 0:
        return None
    return np.mean(pts, axis=0)


# ------------------------------------------------------
# Tâm ban đầu của K-means = các centroid của core-clusters (step 7)
# ------------------------------------------------------
def compute_initial_centroids(core_clusters):
    centroids = []
    for cluster in core_clusters:
        # gộp toàn bộ grid points lại
        all_pts = []
        for g in cluster:
            all_pts.extend(g["points"])
        if len(all_pts) == 0:
            continue
        all_pts = np.array(all_pts)
        centroids.append(np.mean(all_pts, axis=0))
    return np.array(centroids)


# ------------------------------------------------------
# Vẽ lưới core-grid và các core-cluster (minh hoạ Step 4)
# ------------------------------------------------------
def plot_core_groups(points, grid_list, core_clusters, title_prefix="Step 4: Core Grids & Clusters"):
    """Vẽ các ô lưới (grid_list) và highlight các core-grid + cluster.

    Giả định dữ liệu 2D, dùng min_bin[0/1], max_bin[0/1] để vẽ hình chữ nhật.
    """

    # Tính bounds theo toàn bộ grid
    xs_min = [g["min_bin"][0] for g in grid_list]
    xs_max = [g["max_bin"][0] for g in grid_list]
    ys_min = [g["min_bin"][1] for g in grid_list]
    ys_max = [g["max_bin"][1] for g in grid_list]

    xmin, xmax = min(xs_min), max(xs_max)
    ymin, ymax = min(ys_min), max(ys_max)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Bảng màu giống Step 2/3 cho grid
    grid_colors = {
        "empty": ("#f0f0f0", 0.3),   # xám nhạt
        "sparse": ("#87CEEB", 0.4),  # xanh dương nhạt
        "dense": ("#00FF00", 0.9),   # xanh lá
        "core": ("#FFFF00", 1.0),    # vàng
    }

    # Vẽ tất cả grid: chỉ phân biệt core (vàng) và non-core (xám nhạt)
    for g in grid_list:
        x0, x1 = g["min_bin"][0], g["max_bin"][0]
        y0, y1 = g["min_bin"][1], g["max_bin"][1]

        is_core = g.get("is_core", False)
        if is_core:
            facecolor, alpha = grid_colors["core"]
            edgecolor = "red"
        else:
            facecolor, alpha = grid_colors["empty"]
            edgecolor = "red"

        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=0.5,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
        )
        ax.add_patch(rect)

    # Vẽ điểm dữ liệu
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, c="blue", s=15, zorder=10, label=f"Data points ({len(points)})")

    # Cài đặt trục/tên: dùng đúng bounds từng trục
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("Trục X", fontsize=11)
    ax.set_ylabel("Trục Y", fontsize=11)

    # Legend giải thích core-grid / non-core grid và điểm dữ liệu
    legend_elements = [
        patches.Patch(facecolor="#FFFF00", edgecolor="red", label="Core grid"),
        patches.Patch(facecolor="#f0f0f0", edgecolor="red", label="Non-core grid"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=6,
                   label=f"Data points ({len(points)})"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        framealpha=0.9,
    )

    ax.set_title(f"{title_prefix}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle="--")

    # Chừa lề phải cho legend để tránh cảnh báo tight_layout
    plt.subplots_adjust(right=0.8)
    plt.show()


if __name__ == "__main__":
    """Demo Step 4: lần lượt hiển thị Step 1, Step 2, Step 3 rồi tới Step 4.

    Đây chỉ là demo trực tiếp khi chạy file step4_core_grouping.py.
    """

    from step1_compute_M_R import step1_compute_original
    from utils import load_data_txt
    from step2_grid_classification import build_grid, classify_grids, plot_classification
    from step3_recursive_partitioning import step3_handle_dense_grids

    data_path = "data.txt"

    # 1) Step 1: tìm M, R (hàm này đã tự vẽ lưới Step 1)
    step1_result = step1_compute_original(data_path, K=3, max_M=200)
    M = step1_result["M"]
    R = step1_result["R"]

    # 2) Đọc dữ liệu và xây lưới tĩnh (Step 2)
    points = load_data_txt(data_path)
    grid, bounds = build_grid(points, M)
    classified = classify_grids(grid, R)

    # Hiển thị kết quả Step 2
    plot_classification(points, grid, classified, bounds, M, R)

    # 3) Chạy Step 3: phân chia đệ quy các ô dense, có vẽ biểu đồ bên trong
    step3_result = step3_handle_dense_grids(points, M, R, bounds, visualize=True)

    # 4) Chuyển grid Step 2 sang format dùng cho Step 4 (gom core-grids trên lưới tĩnh)
    #    Lưu ý: sau khi refactor Step 2, các ô lưới là đối tượng GridCell, không còn là dict.
    grid_list = []
    for (ix, iy), cell in grid.items():
        # Lấy loại ô từ thuộc tính grid_type (có thể chưa được gán, thì dùng 'unclassified')
        gtype = getattr(cell, "grid_type", "unclassified")
        # Một ô được coi là core nếu grid_type == 'core' hoặc nằm trong danh sách core của Step 2
        is_core = gtype == "core" or cell in classified.get("core", [])

        # Đưa về format dict dùng chung cho Step 4/5: lưu toạ độ biên và danh sách điểm
        grid_list.append(
            {
                "ix": ix,
                "iy": iy,
                "min_bin": (cell.xmin, cell.ymin),
                "max_bin": (cell.xmax, cell.ymax),
                "points": list(getattr(cell, "points", [])),
                "is_core": is_core,
            }
        )

    # 5) Tính và in khoảng cách giữa từng cặp core-grid theo Definition 9,10
    core_entries = [g for g in grid_list if g.get("is_core", False)]
    print("\n===== KHOẢNG CÁCH GIỮA CÁC CORE-GRID (STEP 4) =====")
    if len(core_entries) < 2:
        print("Không đủ core-grid để tính khoảng cách cặp.")
    else:
        for i in range(len(core_entries)):
            for j in range(i + 1, len(core_entries)):
                g1 = core_entries[i]
                g2 = core_entries[j]
                dx = axis_distance(g1, g2, 0)
                dy = axis_distance(g1, g2, 1)
                dist = dx + dy
                adj = dist < 1.0
                print(
                    f"Core ({g1['ix']},{g1['iy']}) - ({g2['ix']},{g2['iy']}): "
                    f"dx={dx:.3f}, dy={dy:.3f}, dist={dist:.3f} -> {'ADJACENT' if adj else 'NOT adjacent'}"
                )

    # 6) Gom core-grids thành các core-cluster và vẽ (Step 4)
    dim = 2
    core_clusters = build_core_clusters(grid_list, dim)

    # In thống kê Step 4 ra terminal
    core_grids_count = sum(1 for g in grid_list if g.get("is_core", False))
    print("\n===== KẾT QUẢ STEP 4: CORE-GROUPING =====")
    print(f"Tổng số grid (Step 2): {len(grid_list)}")
    print(f"Số core-grids (Step 2): {core_grids_count}")
    print(f"Số core-cluster (Step 4): {len(core_clusters)}")
    for idx, cluster in enumerate(core_clusters, start=1):
        print(f"  Cluster {idx}: {len(cluster)} core-grids")

    plot_core_groups(points, grid_list, core_clusters,
                     title_prefix="Step 4 Demo: Core grids & core-clusters từ Step 2")
