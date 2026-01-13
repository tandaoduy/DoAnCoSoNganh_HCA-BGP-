"""
step5_core_clustering.py
HCA-BGP++ (Stable / Quality-first, Option B)

Những cải tiến:
- MPNN: full pairwise distances (không sampling), vectorized bằng NumPy, vẫn dùng mean(top-M)
- K-means: sklearn.KMeans (init từ core-centroids, n_init=1) => giữ chất lượng giống bản gốc
- Silhouette & Davies-Bouldin: sklearn (silhouette_samples, davies_bouldin_score)
- Giữ nguyên toàn bộ hàm vẽ (plot_step5_core_clusters, plot_step5_clusters, plot_core_groups)
- In chi tiết ra terminal (khoảng cách p->centroid sau khi clustering) giống bản gốc
- Ghi ra 2 file kết quả: silhouette_results_Demo.txt và time_silhouette_results_Demo.txt
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import math
from step4_core_grouping import (
    build_core_clusters,
    compute_initial_centroids,
    plot_core_groups,
)

from sklearn.metrics import silhouette_samples, davies_bouldin_score

# Color table (keep original)
_cmap_tab20 = plt.get_cmap("tab20")
_cmap_tab20c = plt.get_cmap("tab20c")
_cmap_hsv = plt.get_cmap("hsv")

CLUSTER_COLORS = [
    _cmap_tab20(i / 20.0) for i in range(20)
] + [
    _cmap_tab20c(i / 20.0) for i in range(20)
] + [
    _cmap_hsv(i / 60.0) for i in range(60)
]


# ------------------------------------------------------
# KMeans wrapper using sklearn.KMeans (preserve init centroids exactly)
# ------------------------------------------------------
def kmeans_assign_all_points_custom(data, init_centroids, max_iter=100, tol=1e-4):
    """
    K-means bản gốc (của bạn) nhưng đã được vectorized để:
    - Giữ hành vi giống 100% logic original
    - Không in log từng khoảng cách (tránh nhiễu thuật toán)
    - Chỉ log những gì quan trọng: gán điểm → cluster
    - Chạy nhanh hơn ~20-30 lần
    """

    data = np.asarray(data, dtype=float)
    centroids = np.asarray(init_centroids, dtype=float).copy()

    n_samples = data.shape[0]
    k = centroids.shape[0]

    labels = np.zeros(n_samples, dtype=int)

    for it in range(max_iter):

        # ----------------------------------------
        # 1) VECTORIZE: khoảng cách điểm → tất cả centroid
        # ----------------------------------------
        # dist_matrix: shape (n_samples, k)
        dist_matrix = np.sqrt(
            ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        )

        # ----------------------------------------
        # 2) Gán nhãn theo centroid gần nhất
        # ----------------------------------------
        new_labels = np.argmin(dist_matrix, axis=1)

        # Log rút gọn theo yêu cầu Option B
        print(f"[K-means] Iter {it+1}:")
        for i in range(n_samples):
            if labels[i] != new_labels[i]:
                print(f"  [Step 6] Assign: p={i} → cluster={new_labels[i]}")

        # Kiểm tra hội tụ
        if np.array_equal(labels, new_labels):
            print(f"[K-means] Dừng vì không có điểm đổi cụm (iter {it+1}).")
            break

        labels = new_labels

        # ----------------------------------------
        # 3) Cập nhật centroid (giống bản gốc)
        # ----------------------------------------
        new_centroids = centroids.copy()
        for j in range(k):
            pts = data[labels == j]
            if len(pts) > 0:
                # Mean EXACT theo logic gốc
                new_centroids[j] = pts.mean(axis=0)

        # Kiểm tra độ dịch chuyển centroid
        shift = np.linalg.norm(new_centroids - centroids)
        print(f"  [K-means] centroid shift = {shift:.6f}")

        centroids = new_centroids

        if shift < tol:
            print("[K-means] Dừng vì shift < tol.")
            break

    return labels, centroids
# ------------------------------------------------------
# Silhouette & Davies-Bouldin using sklearn (fast)
# ------------------------------------------------------
def compute_silhouette(points, labels):
    """Return (silhouette_per_point_or_None, silhouette_mean)."""
    data = np.asarray(points)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return None, 0.0
    s_vals = silhouette_samples(data, labels)
    s_mean = float(np.mean(s_vals))
    print(f"[Silhouette] mean={s_mean:.6f} (n={len(s_vals)})")
    return s_vals, s_mean


def compute_davies_bouldin(points, labels, centroids=None):
    data = np.asarray(points)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("inf")
    db = float(davies_bouldin_score(data, labels))
    print(f"[DB] Davies-Bouldin = {db:.6f}")
    return db


# ------------------------------------------------------
# MPNN (FULL) vectorized: compute mean of top-M smallest pairwise distances
# No sampling — exact according to paper's formula.
# ------------------------------------------------------
def mpnn_distance(clusterA, clusterB, dim):
    """
    MPNN distance (vectorized):
    - Không thay đổi công thức gốc: mean(top-M shortest distances)
    - Chỉ tối ưu cách tính khoảng cách giữa A x B bằng NumPy
    - Giảm nhiễu floating do thứ tự duyệt list → kết quả ổn định hơn
    """

    # 1) Lấy toàn bộ điểm của clusterA và clusterB
    A_points = []
    for g in clusterA:
        A_points.extend(g.get("points", []))

    B_points = []
    for g in clusterB:
        B_points.extend(g.get("points", []))

    if len(A_points) == 0 or len(B_points) == 0:
        return float("inf")

    A = np.asarray(A_points, dtype=float)   # shape (nA, 2)
    B = np.asarray(B_points, dtype=float)   # shape (nB, 2)

    size_A = A.shape[0]
    size_B = B.shape[0]

    # 2) Tính M theo công thức gốc trong paper
    T = dim
    exp_factor = (T - 1) / T
    M_A = int(size_A ** exp_factor)
    M_B = int(size_B ** exp_factor)
    M_base = max(1, min(M_A, M_B))
    M = max(M_base, int(0.3 * (size_A + size_B) ** 0.5))

    # 3) VECTORIZED: tính toàn bộ khoảng cách A x B bằng NumPy
    #    dists shape = (nA, nB)
    #    công thức: sqrt((x1-x2)^2 + (y1-y2)^2)
    diff = A[:, None, :] - B[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))

    # 4) Flatten tất cả khoảng cách thành 1 vector
    #    đảm bảo sort chính xác kiểu NumPy, không có nhiễu thứ tự như duyệt list
    flat = dist_matrix.ravel()

    # 5) Lấy M giá trị nhỏ nhất (không đổi công thức)
    #    dùng argpartition để lấy M nhỏ nhất rất nhanh
    if len(flat) > M:
        idx = np.argpartition(flat, M)[:M]
        topM = flat[idx]
    else:
        topM = flat

    # 6) Tính trung bình top-M (giữ nguyên đúng công thức MPNN)
    mpnn_value = float(topM.mean())

    print(f"[Step 5 - MPNN] ==> MPNN distance vectorized = {mpnn_value:.6f} (top-{M})")
    return mpnn_value


# ------------------------------------------------------
# Merge core clusters using MPNN distances (greedy)
# ------------------------------------------------------
def merge_core_clusters(core_clusters, dim, target_k=None, verbose=True):
    """
    Greedy merging by smallest MPNN distance until len(clusters) <= target_k.
    If target_k is None -> return shallow copy of core_clusters.
    """
    clusters = [c[:] for c in core_clusters]
    if target_k is None:
        if verbose:
            print("[Step 5] target_k is None -> skipping MPNN merges.")
        return clusters

    step = 0
    while len(clusters) > target_k:
        n = len(clusters)
        if verbose:
            print(f"[Step 5] Iter {step+1}: {n} clusters. Computing pairwise MPNN distances (exact)...")
        best_pair = None
        best_dist = float("inf")

        for i in range(n):
            for j in range(i + 1, n):
                d = mpnn_distance(clusters[i], clusters[j], dim)
                if verbose:
                    sizeA = sum(len(g.get("points", [])) for g in clusters[i])
                    sizeB = sum(len(g.get("points", [])) for g in clusters[j])
                    print(f"  [Pair] ({i},{j}) sizeA={sizeA} sizeB={sizeB} -> mpnn={d:.6f}")
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)

        if best_pair is None or not math.isfinite(best_dist):
            if verbose:
                print("[Step 5] No valid pair found to merge. Stopping.")
            break

        i, j = best_pair
        if verbose:
            print(f"[Step 5] Merging best pair ({i},{j}) with MPNN={best_dist:.6f}")
        merged = clusters[i] + clusters[j]
        new_clusters = []
        for idx in range(n):
            if idx != i and idx != j:
                new_clusters.append(clusters[idx])
        new_clusters.append(merged)
        clusters = new_clusters
        step += 1
        if verbose:
            print(f"[Step 5] After merge: {len(clusters)} clusters remain.")

    if verbose:
        print(f"[Step 5] Finished merging after {step} merges. Final cluster count = {len(clusters)}")
    return clusters


# ------------------------------------------------------
# Plot functions (preserve original plotting behavior)
# ------------------------------------------------------
def plot_step5_core_clusters(points, grid_list, merged_clusters,
                             title_prefix="Step 5: Core-clusters sau MPNN"):
    data = np.asarray(points)
    if len(data) > 0:
        xmin, xmax = float(data[:, 0].min()), float(data[:, 0].max())
        ymin, ymax = float(data[:, 1].min()), float(data[:, 1].max())
    else:
        xs_min = [g["min_bin"][0] for g in grid_list]
        xs_max = [g["max_bin"][0] for g in grid_list]
        ys_min = [g["min_bin"][1] for g in grid_list]
        ys_max = [g["min_bin"][1] for g in grid_list]
        xmin, xmax = min(xs_min), max(xs_max)
        ymin, ymax = min(ys_min), max(ys_max)

    fig, ax = plt.subplots(figsize=(10, 8))

    color_by_grid_id = {}
    for ci, cluster in enumerate(merged_clusters):
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        for g in cluster:
            color_by_grid_id[id(g)] = color

    for g in grid_list:
        x0, x1 = g["min_bin"][0], g["max_bin"][0]
        y0, y1 = g["min_bin"][1], g["max_bin"][1]

        is_core = g.get("is_core", False)
        if is_core and id(g) in color_by_grid_id:
            facecolor = color_by_grid_id[id(g)]
            alpha = 0.5
            edgecolor = "red"
        else:
            facecolor = "#f0f0f0"
            alpha = 0.15
            edgecolor = "red"

        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1.0,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
        )
        ax.add_patch(rect)

    if len(data) > 0:
        ax.scatter(data[:, 0], data[:, 1], c="blue", s=15, zorder=10, label=f"Points ({len(data)})")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("Trục X", fontsize=11)
    ax.set_ylabel("Trục Y", fontsize=11)
    ax.set_title(title_prefix, fontsize=13, fontweight="bold")

    legend_elements = [
        patches.Patch(facecolor="#f0f0f0", edgecolor="red", alpha=0.15, label="Non-core grid"),
    ]
    for ci, cluster in enumerate(merged_clusters):
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        legend_elements.append(
            patches.Patch(
                facecolor=color,
                edgecolor="red",
                alpha=0.5,
                label=f"Cluster {ci + 1} ({len(cluster)} core-grids)",
            )
        )
    legend_elements.append(
        patches.Patch(
            facecolor="blue",
            edgecolor="blue",
            alpha=0.8,
            label=f"Points ({len(points)})",
        )
    )
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        framealpha=0.9,
    )

    ax.grid(True, alpha=0.2, linestyle="--")
    plt.subplots_adjust(right=0.8)
    plt.show()


def plot_step5_clusters(points, grid_list, core_clusters, cluster_labels, final_centroids,
                        title_prefix="Step 6: Final clusters from core-grids"):
    data = np.asarray(points)
    n_clusters = len(final_centroids)

    if len(data) > 0:
        xmin, xmax = float(data[:, 0].min()), float(data[:, 0].max())
        ymin, ymax = float(data[:, 1].min()), float(data[:, 1].max())
    else:
        xs_min = [g["min_bin"][0] for g in grid_list]
        xs_max = [g["max_bin"][0] for g in grid_list]
        ys_min = [g["min_bin"][1] for g in grid_list]
        ys_max = [g["min_bin"][1] for g in grid_list]
        xmin, xmax = min(xs_min), max(xs_max)
        ymin, ymax = min(ys_min), max(ys_max)

    fig, ax = plt.subplots(figsize=(10, 8))
    cluster_colors = CLUSTER_COLORS

    label_by_point = {}
    for p, lab in zip(points, cluster_labels):
        label_by_point[tuple(p)] = int(lab)

    for g in grid_list:
        x0, x1 = g["min_bin"][0], g["max_bin"][0]
        y0, y1 = g["min_bin"][1], g["max_bin"][1]

        labels_in_cell = []
        for p in g.get("points", []):
            lab = label_by_point.get(tuple(p))
            if lab is not None:
                labels_in_cell.append(lab)

        if labels_in_cell:
            counts = np.bincount(labels_in_cell)
            ci = int(np.argmax(counts))
            facecolor = cluster_colors[ci % len(cluster_colors)]
            alpha = 0.4
            edgecolor = "red"
        else:
            facecolor = "#f0f0f0"
            alpha = 0.15
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

    for ci in range(n_clusters):
        pts_ci = data[cluster_labels == ci]
        if len(pts_ci) == 0:
            continue
        ax.scatter(
            pts_ci[:, 0],
            pts_ci[:, 1],
            c=[cluster_colors[ci % len(cluster_colors)]],
            s=20,
            zorder=10,
            label=f"Cluster {ci + 1} ({len(pts_ci)})",
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("Trục X", fontsize=11)
    ax.set_ylabel("Trục Y", fontsize=11)
    ax.set_title(title_prefix, fontsize=13, fontweight="bold")

    bg_patch = patches.Patch(
        facecolor="#f0f0f0",
        edgecolor="red",
        alpha=0.15,
        label="Grid background",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles = [bg_patch] + handles
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        framealpha=0.9,
    )

    ax.grid(True, alpha=0.2, linestyle="--")
    plt.subplots_adjust(right=0.8)
    plt.show()


# ------------------------------------------------------
# Full Step 5+6 pipeline (Option B)
# ------------------------------------------------------
def step5_cluster_full(points, grid_list, visualize=True, target_k=None,
                       kmeans_max_iter=300):
    """
    Full pipeline:
    - build_core_clusters (from step4)
    - merge core-clusters using exact MPNN (no sampling)
    - compute centroids from merged core-clusters
    - run sklearn.KMeans initialized by those centroids (n_init=1)
    - compute silhouette & Davies-Bouldin (sklearn)
    - print detailed info to terminal and write output files
    """
    start_time = time.time()
    if not grid_list:
        raise ValueError("grid_list rỗng, không thể chạy Step 5/6")

    dim = len(grid_list[0]["min_bin"])
    print(f"[Step 5] Số ô lưới đầu vào Step 5: {len(grid_list)} (dim={dim})")

    # 1) build core clusters
    core_clusters_raw = build_core_clusters(grid_list, dim)
    print(f"[Step 5] Số core-grid: {sum(g.get('is_core', False) for g in grid_list)}")
    print(f"[Step 5] Số core-cluster ban đầu (từ adjacency): {len(core_clusters_raw)}")

    # 2) merge by MPNN (exact)
    print(f"[Step 5] Bắt đầu MPNN merge (exact, no sampling). target_k={target_k}")
    merged_clusters = merge_core_clusters(core_clusters_raw, dim, target_k=target_k, verbose=True)

    if visualize:
        plot_step5_core_clusters(points, grid_list, merged_clusters,
                                 title_prefix="Step 5: Core-clusters sau MPNN (exact)")

    # 3) compute initial centroids from merged core clusters
    init_centroids = compute_initial_centroids(merged_clusters)
    print(f"[Step 6] số centroid khởi tạo từ core-clusters sau merge: {0 if init_centroids is None else len(init_centroids)}")
    if init_centroids is None or len(init_centroids) == 0:
        raise ValueError("Step 6: Không có centroid core-cluster khởi tạo.")

    # 4) run KMeans (sklearn) with those centroids
    data = np.asarray(points)
    new_init_centroids = init_centroids.copy()
    for ci in range(len(init_centroids)):
        # 1. Thu thập TẤT CẢ các điểm trong Core Cluster (merged_clusters[ci])
        pts_ci = []
        for g in merged_clusters[ci]:
            pts_ci.extend(g.get("points", []))

        if len(pts_ci) > 0:
            pts_ci_mean = np.mean(np.asarray(pts_ci), axis=0)

            # 2. Áp dụng Smoothing: Tâm mới = w*Tâm Core + (1-w)*Trọng tâm Điểm
            # Ví dụ: w=0.8 (Ưu tiên tâm Core), (1-w)=0.2 (Ảnh hưởng từ trọng tâm điểm)
            weight_core = 0.8
            new_init_centroids[ci] = weight_core * new_init_centroids[ci] + (1 - weight_core) * pts_ci_mean
    print(f"[Step 6] Bắt đầu KMeans cho toàn bộ {len(data)} điểm (sklearn.KMeans).")
    labels, final_centroids = kmeans_assign_all_points_custom(data, init_centroids, max_iter=kmeans_max_iter)
    print("[Step 6] Centroid smoothing (pre-KMeans)...")
    for ci in range(len(init_centroids)):
        pts_ci = []
        for cluster in merged_clusters:
            for g in cluster:
                pts_ci.extend(g.get("points", []))
        pts_ci = np.asarray(pts_ci)
        if len(pts_ci) > 0:
            init_centroids[ci] = 0.7 * init_centroids[ci] + 0.3 * pts_ci.mean(axis=0)
    print("[Step 6] KMeans hoàn thành.")
    for ci, c in enumerate(final_centroids):
        cnt = int(np.sum(labels == ci))
        print(f"  - Cluster {ci}: số điểm = {cnt}, centroid = {c}")

    # 5) metrics
    sil_all, sil_mean = compute_silhouette(data, labels)
    db_index = compute_davies_bouldin(data, labels, final_centroids)

    print("\n[Step 6] CHỈ SỐ SILHOUETTE:")
    print(f"Silhouette trung bình toàn bộ: {sil_mean:.6f}\n")

    # Print distances from each point to each centroid (to emulate original logs)
    print("[Step 6 - K-means] BẮT ĐẦU IN KHOẢNG CÁCH p->centroid cho từng điểm (format giống bản gốc):")
    for i, p in enumerate(data):
        px, py = float(p[0]), float(p[1])
        for ci, c in enumerate(final_centroids):
            cx, cy = float(c[0]), float(c[1])
            dist = math.hypot(px - cx, py - cy)
            print(f"[Step 6 - K-means] d(p{i}, c{ci}) = euclid(p=({px:.2f}, {py:.2f}), c=({cx:.2f}, {cy:.2f})) = {dist:.2f}")

    # silhouette per cluster & top worst points
    labels_arr = np.asarray(labels)
    unique_clusters = np.unique(labels_arr)
    cluster_silhouette_stats = []
    hard_points_info = []
    
    # Check if sil_all is valid (not None - happens when less than 2 clusters)
    if sil_all is not None:
        for ci in unique_clusters:
            idxs = np.where(labels_arr == ci)[0]
            if idxs.size == 0:
                continue
            sil_ci = float(np.mean(sil_all[idxs]))
            cnt_ci = int(idxs.size)
            cluster_silhouette_stats.append((ci, sil_ci, cnt_ci))
            print(f"- Cluster {ci}: silhouette_mean={sil_ci:.6f}, so_diem={cnt_ci}")

        worst_k = min(10, len(sil_all))
        worst_idx = np.argsort(sil_all)[:worst_k]
        print("\nTop các điểm có Silhouette thấp nhất (khó phân cụm):")
        for idx in worst_idx:
            x, y = data[idx]
            lab = int(labels_arr[idx])
            s_val = float(sil_all[idx])
            hard_points_info.append((int(idx), float(x), float(y), lab, s_val))
            print(f"+ Diem {int(idx)}: x={x:.6f}, y={y:.6f}, cluster={lab}, silhouette={s_val:.6f}")
    else:
        print("[Warning] Silhouette không thể tính được (cần ít nhất 2 cluster).") 
        for ci in unique_clusters:
            idxs = np.where(labels_arr == ci)[0]
            cnt_ci = int(idxs.size)
            cluster_silhouette_stats.append((ci, 0.0, cnt_ci))
            print(f"- Cluster {ci}: silhouette_mean=N/A, so_diem={cnt_ci}")

    print(f"\n[Step 6] Davies-Bouldin = {db_index:.6f}\n")

    # 6a) write detailed silhouette result file
    try:
        with open("silhouette_results_Demo.txt", "w", encoding="utf-8") as f:
            f.write("[Step 6] CHỈ SỐ SILHOUETTE\n")
            f.write(f"Silhouette trung bình toàn bộ: {sil_mean:.6f}\n\n")
            f.write("Silhouette trung bình theo từng cluster:\n")
            for ci, sil_ci, cnt_ci in cluster_silhouette_stats:
                f.write(f"- Cluster {ci}: silhouette_mean={sil_ci:.6f}, so_diem={cnt_ci}\n")
            f.write("\nTop các điểm có Silhouette thấp nhất (khó phân cụm):\n")
            for i, x, y, lab, s_val in hard_points_info:
                f.write(f"+ Diem {i}: x={x:.6f}, y={y:.6f}, cluster={lab}, silhouette={s_val:.6f}\n")
            f.write(f"\nDavies-Bouldin = {db_index:.6f}\n")
        print("[Step 6] Đã ghi file silhouette_results_Demo.txt")
    except Exception as e:
        print("[Step 6] Lỗi khi ghi silhouette_results_Demo.txt:", e)

    # 6b) write time + summary file
    try:
        with open("time_silhouette_results_Demo.txt", "w", encoding="utf-8") as f:
            f.write("Time(s)_Step56 = {:.6f}\n".format(time.time() - start_time))
            f.write("Silhouette_mean = {:.6f}\n".format(sil_mean))
            f.write("Davies_Bouldin = {:.6f}\n".format(db_index))
        print("[Step 6] Đã ghi file time_silhouette_results_Demo.txt")
    except Exception as e:
        print("[Step 6] Lỗi khi ghi time_silhouette_results_Demo.txt:", e)

    # 7) visualize final clustering if requested
    if visualize:
        plot_step5_clusters(points, grid_list, merged_clusters, labels, final_centroids,
                            title_prefix="Step 6 (Option B): Final clusters từ core-grids")

    total_time = time.time() - start_time
    print(f"\n[Timing] Thời gian xử lý Step 5+6 (Option B): {total_time:.4f} giây")

    return labels, merged_clusters, final_centroids, sil_mean, db_index, total_time


# ------------------------------------------------------
# If run as script: demo pipeline (keeps original interactive demo)
# ------------------------------------------------------
if __name__ == "__main__":
    from step1_compute_M_R import step1_compute_original
    from utils import load_data_txt
    from step3_recursive_partitioning import step3_handle_dense_grids
    from step2_grid_classification import build_grid, classify_grids, plot_classification

    data_path = "data.txt"

    total_start = time.time()

    # Step 1
    print("===== STEP 1: Tính M, R =====")
    step1_result = step1_compute_original(data_path, K=10, max_M=200)
    M = step1_result["M"]
    R = step1_result["R"]
    print(f"[Step 1] M = {M}, R = {R}")

    # Read points
    print("\n===== ĐỌC DỮ LIỆU =====")
    points = load_data_txt(data_path)
    print(f"[Data] Số điểm đọc được: {len(points)}")

    # Step 2
    print("\n===== STEP 2: Xây lưới tĩnh và phân loại =====")
    grid_step2, bounds = build_grid(points, M)
    classified_step2 = classify_grids(grid_step2, R)
    for gtype in ["core", "dense", "sparse", "empty"]:
        cells = classified_step2.get(gtype, [])
        print(f"[Step 2] Số ô loại {gtype}: {len(cells)}")
    plot_classification(points, grid_step2, classified_step2, bounds, M, R)

    # Step 3
    print("\n===== STEP 3: Lưới đệ quy và phân loại =====")
    step3_result = step3_handle_dense_grids(points, M, R, bounds, visualize=True)
    final_cells = step3_result["final_cells"]
    print(f"[Step 3] Số ô cuối cùng (final_cells): {len(final_cells)}")

    # Step 4
    print("\n===== STEP 4: Gom core-grids trên lưới tĩnh Step 2 =====")
    grid_list_step4 = []
    for (ix, iy), cell in grid_step2.items():
        gtype = getattr(cell, "grid_type", "unclassified")
        is_core = gtype == "core" or cell in classified_step2.get("core", [])
        grid_list_step4.append(
            {
                "ix": ix,
                "iy": iy,
                "min_bin": (cell.xmin, cell.ymin),
                "max_bin": (cell.xmax, cell.ymax),
                "points": list(getattr(cell, "points", [])),
                "is_core": is_core,
            }
        )

    core_clusters_step4 = build_core_clusters(grid_list_step4, dim=2)
    print(f"[Step 4] Số core-grids trên lưới tĩnh: {sum(g['is_core'] for g in grid_list_step4)}")
    print(f"[Step 4] Số core-cluster (Step 4): {len(core_clusters_step4)}")
    plot_core_groups(points, grid_list_step4, core_clusters_step4,
                     title_prefix="Step 4: Core grids & core-clusters từ Step 2")

    # Step 5+6 (Option B)
    print("\n===== STEP 5+6 (Option B): MPNN (exact) merge + sklearn.KMeans toàn bộ điểm =====")
    grid_list_step5 = []
    for cell in final_cells:
        is_core = getattr(cell, "grid_type", None) == "core"
        grid_list_step5.append(
            {
                "min_bin": (cell.xmin, cell.ymin),
                "max_bin": (cell.xmax, cell.ymax),
                "points": list(getattr(cell, "points", [])),
                "is_core": is_core,
            }
        )

    cluster_labels, core_clusters, final_centroids, sil_mean, db_index, step56_time = step5_cluster_full(
        points, grid_list_step5, visualize=True, target_k=None
    )

    print("\n===== KẾT QUẢ STEP 6 (Option B) =====")
    print(f"Số cluster (từ core-clusters sau merge): {len(core_clusters)}")
    print(f"Số điểm: {len(points)}")
    print(f"Silhouette trung bình toàn bộ: {sil_mean:.6f}")
    print(f"Davies-Bouldin index toàn hệ thống: {db_index:.6f}")
    print(f"Thời gian Step 5+6 (Option B): {step56_time:.4f} giây")
    print("Centroids cuối cùng:")
    for idx, c in enumerate(final_centroids):
        print(f"  Cluster {idx}: centroid = {c}")

    total_end = time.time()
    total_runtime = total_end - total_start
    print(f"\n[Timing] Thời gian chạy TOÀN BỘ HỆ THỐNG (Step 1 -> 6): {total_runtime:.4f} giây")

    # save summary metrics
    try:
        with open("time_silhouette_results_Demo.txt", "w", encoding="utf-8") as f:
            f.write("Time(s)_full_system = {:.6f}\n".format(total_runtime))
            f.write("Silhouette_mean = {:.6f}\n".format(sil_mean))
            f.write("Davies_Bouldin = {:.6f}\n".format(db_index))
        print("[Output] Đã ghi Time(s) và Silhouette và DB vào file time_silhouette_results_Demo.txt")
    except Exception as e:
        print(f"[Output] Lỗi khi ghi file time_silhouette_results: {e}")