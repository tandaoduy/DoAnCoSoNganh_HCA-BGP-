"""Step 5/6: core-cluster merging (MPNN) and final K-means clustering.

This module implements the MPNN-based merging of core-clusters (Step 5)
and a custom K-means run initialized from core-cluster centroids (Step 6).
It also computes clustering quality metrics such as Silhouette and
Davies-Bouldin indices.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import math
from utils import euclid
from step4_core_grouping import (
    build_core_clusters,
    compute_initial_centroids,
    plot_core_groups,
)

# Bảng màu dùng chung cho tối đa ~100 cluster
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
# Phân cụm bằng K-means với tâm khởi tạo = core-centroids
# (Cài tay, dùng utils.euclid làm metric)
# ------------------------------------------------------
def kmeans_assign_all_points_custom(data, init_centroids, max_iter=100, tol=1e-4):
    data = np.asarray(data)
    centroids = np.asarray(init_centroids, dtype=float).copy()
    n_samples = data.shape[0]
    k = centroids.shape[0]

    # Nhãn cụm cho từng điểm
    labels = np.zeros(n_samples, dtype=int)
    n_iter = 0

    for it in range(max_iter):
        # 1) Gán từng điểm vào centroid gần nhất (dùng euclid từ utils)
        changed = False
        for i in range(n_samples):
            p = data[i]
            # tính khoảng cách tới từng centroid
            dists = [euclid(p, centroids[j]) for j in range(k)]
            for j in range(k):
                c = centroids[j]
                d = dists[j]

                # Định dạng toạ độ điểm p và centroid c với 2 chữ số thập phân để log gọn hơn
                px, py = float(p[0]), float(p[1])
                cx, cy = float(c[0]), float(c[1])
                print(
                    f"[Step 6 - K-means] d(p{i}, c{j}) = "
                    f"euclid(p=({px:.2f}, {py:.2f}), c=({cx:.2f}, {cy:.2f})) = {d:.2f}"
                )
            best_label = int(np.argmin(dists))
            if labels[i] != best_label:
                labels[i] = best_label
                changed = True

        # Nếu không điểm nào đổi cụm nữa thì coi như hội tụ
        if not changed:
            print(f"[K-means] Dừng sớm vì không có điểm nào đổi cụm. Iter {it + 1}")
            break

        # 2) Cập nhật lại centroid từng cụm
        new_centroids = centroids.copy()
        for j in range(k):
            cluster_points = data[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)

        # Kiểm tra độ dịch chuyển của centroids để dừng sớm
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        # In thông tin từng vòng lặp K-means
        cluster_sizes = [int(np.sum(labels == j)) for j in range(k)]
        n_iter = it + 1
        print(f"[K-means] Iter {n_iter}: shift={shift:.2f}, sizes={cluster_sizes}")
        if shift < tol:
            print("[K-means] Dừng sớm vì shift < tol.")
            break

    print(f"[K-means] Kết thúc sau {n_iter} vòng lặp.")
    return labels, centroids


# ------------------------------------------------------
# Tính Silhouette cho toàn bộ điểm (dùng khoảng cách euclid)
# ------------------------------------------------------
def compute_silhouette(points, labels):
    """Trả về (silhouette_từng_điểm, silhouette_trung_bình)."""
    data = np.asarray(points)
    labels = np.asarray(labels)
    n = len(data)
    if n == 0:
        return np.array([]), 0.0

    unique_clusters = np.unique(labels)
    if len(unique_clusters) < 2:
        # Silhouette không có ý nghĩa nếu chỉ có 1 cụm
        return np.zeros(n, dtype=float), 0.0

    # Ma trận khoảng cách n x n
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = euclid(data[i], data[j])
            D[i, j] = D[j, i] = d

    s = np.zeros(n, dtype=float)
    indices = np.arange(n)
    for i in range(n):
        ci = labels[i]
        same = (labels == ci)
        # Loại bỏ chính điểm i khỏi mask cùng cụm
        same_no_i = np.logical_and(same, indices != i)
        # Nếu cụm chỉ có 1 điểm -> silhouette = 0
        if not np.any(same_no_i):
            s[i] = 0.0
            continue

        # a(i): khoảng cách trung bình tới các điểm cùng cụm
        a_i = D[i, same_no_i].mean()

        # b(i): khoảng cách trung bình nhỏ nhất tới cụm khác
        b_i = float("inf")
        for c in unique_clusters:
            if c == ci:
                continue
            other = (labels == c)
            if not np.any(other):
                continue
            d_c = D[i, other].mean()
            if d_c < b_i:
                b_i = d_c

        s[i] = (b_i - a_i) / max(a_i, b_i) if b_i > 0 else 0.0

    return s, float(s.mean())


def compute_davies_bouldin(points, labels, centroids):
    """Compute Davies-Bouldin index for clustering using Euclidean distance.

    Returns a float (DB index). If there are fewer than 2 non-empty clusters,
    returns float('inf').
    """
    data = np.asarray(points)
    labels = np.asarray(labels)
    centroids = np.asarray(centroids)

    # Identify non-empty clusters (by label index)
    unique_labels = np.unique(labels)
    # If centroids are given as list/array indexed by label, consider labels 0..len(centroids)-1
    # But evaluate only those labels that actually have members
    valid_clusters = [int(l) for l in unique_labels if np.sum(labels == l) > 0]

    if len(valid_clusters) < 2:
        print("[DB] Không đủ cụm (>=2) để tính Davies-Bouldin. Trả về inf.")
        return float("inf")

    # Compute S_i: average distance of points in cluster i to its centroid
    S = {}
    for i in valid_clusters:
        members = data[labels == i]
        if len(members) == 0:
            S[i] = 0.0
        else:
            # use euclid for robust 2D/ND distance
            dists = [euclid(p, centroids[i]) for p in members]
            S[i] = float(np.mean(dists)) if len(dists) > 0 else 0.0

    # Compute pairwise centroid distances and R_ij
    D_i = []
    for i in valid_clusters:
        max_r = 0.0
        for j in valid_clusters:
            if j == i:
                continue
            dist_c = euclid(centroids[i], centroids[j])
            if dist_c <= 0:
                r_ij = float("inf")
            else:
                r_ij = (S[i] + S[j]) / dist_c
            if r_ij > max_r:
                max_r = r_ij
        D_i.append(max_r)

    # DB index is mean of D_i
    db_index = float(np.mean(D_i)) if len(D_i) > 0 else float("inf")
    print(f"[DB] Davies-Bouldin index = {db_index:.6f}")
    return db_index


# =============================
# IN CHI TIẾT CÔNG THỨC TÍNH TOÁN K-MEANS, EUCLIDEAN, SILHOUETTE, DAVIES-BOULDIN
# =============================
def print_kmeans_formulas_detail(points, init_centroids, max_iter=100, tol=1e-4):
    """
    In chi tiết từng bước tính toán K-means với công thức Euclidean,
    rồi tính Silhouette và Davies-Bouldin sau khi hội tụ.
    """
    data = np.asarray(points)
    centroids = np.asarray(init_centroids, dtype=float).copy()
    n_samples = data.shape[0]
    k = centroids.shape[0]
    
    print("\n" + "="*100)
    print("CHI TIẾT CÔNG THỨC TÍNH TOÁN K-MEANS VÀ CÁC CHỈ SỐ ĐÁNH GIÁ")
    print("="*100)
    
    # =============================================
    # PHẦN 1: CÔNG THỨC EUCLIDEAN DISTANCE
    # =============================================
    print("\n" + "─"*100)
    print("📐 CÔNG THỨC KHOẢNG CÁCH EUCLIDEAN (Euclidean Distance)")
    print("─"*100)
    print("\n▶ CÔNG THỨC TỔNG QUÁT:")
    print("   d(p, c) = √[Σᵢ (pᵢ - cᵢ)²]")
    print("   Trong đó:")
    print("   • p = (p₁, p₂, ..., pₙ) là tọa độ điểm dữ liệu")
    print("   • c = (c₁, c₂, ..., cₙ) là tọa độ tâm cụm (centroid)")
    print("   • n là số chiều dữ liệu")
    print("\n▶ VỚI DỮ LIỆU 2 CHIỀU (x, y):")
    print("   d(p, c) = √[(px - cx)² + (py - cy)²]")
    
    # =============================================
    # PHẦN 2: THUẬT TOÁN K-MEANS
    # =============================================
    print("\n" + "─"*100)
    print("🔄 THUẬT TOÁN K-MEANS")
    print("─"*100)
    print("\n▶ BƯỚC 1 - KHỞI TẠO:")
    print(f"   • Số điểm dữ liệu: n = {n_samples}")
    print(f"   • Số cụm mong muốn: K = {k}")
    print("   • Tâm cụm ban đầu (từ Core-Cluster centroids):")
    for j, c in enumerate(centroids):
        print(f"     c{j} = ({c[0]:.4f}, {c[1]:.4f})")
    
    labels = np.zeros(n_samples, dtype=int)
    
    for it in range(max_iter):
        print(f"\n▶ VÒNG LẶP {it + 1}:")
        print("─"*80)
        
        # ========================
        # BƯỚC 2A: GÁN ĐIỂM VÀO CỤM
        # ========================
        print("\n   📌 BƯỚC 2A: GÁN MỖI ĐIỂM VÀO CỤM CÓ TÂM GẦN NHẤT")
        
        changed = False
        for i in range(n_samples):  # In chi tiết tất cả các điểm
            p = data[i]
            print(f"\n   🔹 Điểm {i}: p = ({p[0]:.4f}, {p[1]:.4f})")
            
            dists = []
            for j in range(k):
                c = centroids[j]
                dx = p[0] - c[0]
                dy = p[1] - c[1]
                d = math.sqrt(dx**2 + dy**2)
                dists.append(d)
                
                print(f"      d(p{i}, c{j}) = √[({p[0]:.4f} - {c[0]:.4f})² + ({p[1]:.4f} - {c[1]:.4f})²]")
                print(f"                   = √[{dx**2:.6f} + {dy**2:.6f}]")
                print(f"                   = √{dx**2 + dy**2:.6f}")
                print(f"                   = {d:.6f}")
            
            best_label = int(np.argmin(dists))
            print(f"      → min(d) = {min(dists):.6f} tại c{best_label}")
            print(f"      → Gán điểm {i} vào Cluster {best_label}")
            
            if labels[i] != best_label:
                labels[i] = best_label
                changed = True
        
        # ========================
        # BƯỚC 2B: CẬP NHẬT TÂM CỤM
        # ========================
        print("\n   📌 BƯỚC 2B: CẬP NHẬT TÂM CỤM (CENTROID)")
        print("   Công thức: cⱼ = (1/|Cⱼ|) × Σ(xi), với xi ∈ Cⱼ")
        
        new_centroids = centroids.copy()
        for j in range(k):
            cluster_points = data[labels == j]
            n_j = len(cluster_points)
            
            if n_j > 0:
                sum_x = sum(p[0] for p in cluster_points)
                sum_y = sum(p[1] for p in cluster_points)
                new_cx = sum_x / n_j
                new_cy = sum_y / n_j
                new_centroids[j] = np.array([new_cx, new_cy])
                
                print(f"\n   Cluster {j}: {n_j} điểm")
                print(f"      Tổng X = {sum_x:.4f}, Tổng Y = {sum_y:.4f}")
                print(f"      c{j}_new = (1/{n_j}) × ({sum_x:.4f}, {sum_y:.4f})")
                print(f"             = ({new_cx:.4f}, {new_cy:.4f})")
        
        # Kiểm tra hội tụ
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        
        print(f"\n   📊 TỔNG KẾT VÒNG LẶP {it + 1}:")
        print(f"      • Độ dịch chuyển tâm cụm (shift): {shift:.6f}")
        print(f"      • Ngưỡng hội tụ (tol): {tol}")
        
        cluster_sizes = [int(np.sum(labels == j)) for j in range(k)]
        print(f"      • Phân bố điểm: {cluster_sizes}")
        
        if not changed:
            print(f"\n   ✅ DỪNG: Không có điểm nào đổi cụm → K-means hội tụ!")
            break
        
        if shift < tol:
            print(f"\n   ✅ DỪNG: shift = {shift:.6f} < tol = {tol} → K-means hội tụ!")
            break
    
    print(f"\n▶ KẾT THÚC K-MEANS SAU {it + 1} VÒNG LẶP")
    print("   Tâm cụm cuối cùng:")
    for j, c in enumerate(centroids):
        cnt = int(np.sum(labels == j))
        print(f"   • Cluster {j}: centroid = ({c[0]:.4f}, {c[1]:.4f}), số điểm = {cnt}")
    
    # =============================================
    # PHẦN 3: CHỈ SỐ SILHOUETTE
    # =============================================
    print("\n" + "─"*100)
    print("📊 CHỈ SỐ SILHOUETTE (Silhouette Coefficient)")
    print("─"*100)
    
    print("\n▶ CÔNG THỨC:")
    print("   s(i) = (b(i) - a(i)) / max(a(i), b(i))")
    print("\n   Trong đó:")
    print("   • a(i) = khoảng cách trung bình từ điểm i đến các điểm CÙNG cụm")
    print("   • b(i) = khoảng cách trung bình nhỏ nhất từ điểm i đến cụm KHÁC gần nhất")
    print("   • s(i) ∈ [-1, 1]: -1 = phân cụm sai, 0 = biên, 1 = phân cụm tốt")
    
    # Tính Silhouette cho vài điểm đầu
    unique_clusters = np.unique(labels)
    if len(unique_clusters) >= 2:
        # Ma trận khoảng cách
        D = np.zeros((n_samples, n_samples), dtype=float)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = euclid(data[i], data[j])
                D[i, j] = D[j, i] = d
        
        print("\n▶ TÍNH CHI TIẾT CHO TẤT CẢ CÁC ĐIỂM:")
        s_values = np.zeros(n_samples)
        
        for i in range(n_samples):  # Tính cho tất cả các điểm
            ci = labels[i]
            print(f"\n   🔹 Điểm {i}: p = ({data[i][0]:.4f}, {data[i][1]:.4f}), Cluster = {ci}")
            
            # Tính a(i)
            same = (labels == ci)
            same[i] = False  # loại bỏ chính nó
            if np.any(same):
                dists_same = D[i, same]
                a_i = float(np.mean(dists_same))
                print(f"      a({i}) = trung bình khoảng cách đến {np.sum(same)} điểm cùng cụm")
                print(f"           = {a_i:.6f}")
            else:
                a_i = 0
                print(f"      a({i}) = 0 (chỉ có 1 điểm trong cụm)")
            
            # Tính b(i)
            b_i = float("inf")
            for c in unique_clusters:
                if c == ci:
                    continue
                other = (labels == c)
                if np.any(other):
                    d_c = float(np.mean(D[i, other]))
                    if d_c < b_i:
                        b_i = d_c
            print(f"      b({i}) = khoảng cách trung bình nhỏ nhất đến cụm khác")
            print(f"           = {b_i:.6f}")
            
            # Tính s(i)
            if max(a_i, b_i) > 0:
                s_i = (b_i - a_i) / max(a_i, b_i)
            else:
                s_i = 0
            s_values[i] = s_i
            
            print(f"      s({i}) = ({b_i:.6f} - {a_i:.6f}) / max({a_i:.6f}, {b_i:.6f})")
            print(f"           = {b_i - a_i:.6f} / {max(a_i, b_i):.6f}")
            print(f"           = {s_i:.6f}")
        
        sil_mean = float(np.mean(s_values))
        print(f"\n▶ SILHOUETTE TRUNG BÌNH TOÀN BỘ:")
        print(f"   S_avg = (1/n) × Σ s(i) = {sil_mean:.6f}")
    else:
        sil_mean = 0
        print("\n   ⚠ Chỉ có 1 cụm → Không tính được Silhouette")
    
    # =============================================
    # PHẦN 4: CHỈ SỐ DAVIES-BOULDIN
    # =============================================
    print("\n" + "─"*100)
    print("📊 CHỈ SỐ DAVIES-BOULDIN (Davies-Bouldin Index)")
    print("─"*100)
    
    print("\n▶ CÔNG THỨC:")
    print("   DB = (1/K) × Σ Dᵢ")
    print("\n   Trong đó:")
    print("   • Dᵢ = max(Rᵢⱼ) với j ≠ i")
    print("   • Rᵢⱼ = (Sᵢ + Sⱼ) / d(cᵢ, cⱼ)")
    print("   • Sᵢ = khoảng cách trung bình từ các điểm trong cụm i đến tâm cᵢ")
    print("   • DB càng nhỏ → phân cụm càng tốt")
    
    if len(unique_clusters) >= 2:
        print("\n▶ TÍNH CHI TIẾT:")
        
        # Tính S cho mỗi cụm
        S = {}
        for j in range(k):
            members = data[labels == j]
            if len(members) > 0:
                dists = [euclid(p, centroids[j]) for p in members]
                S[j] = float(np.mean(dists))
            else:
                S[j] = 0
            print(f"\n   S[{j}] = khoảng cách trung bình trong Cluster {j}")
            print(f"        = {S[j]:.6f}")
        
        # Tính D cho mỗi cụm
        D_i = []
        for i in range(k):
            print(f"\n   Tính D[{i}]:")
            max_r = 0
            for j in range(k):
                if j == i:
                    continue
                dist_c = euclid(centroids[i], centroids[j])
                if dist_c > 0:
                    r_ij = (S[i] + S[j]) / dist_c
                else:
                    r_ij = float("inf")
                print(f"      R[{i},{j}] = ({S[i]:.6f} + {S[j]:.6f}) / {dist_c:.6f} = {r_ij:.6f}")
                if r_ij > max_r:
                    max_r = r_ij
            D_i.append(max_r)
            print(f"      D[{i}] = max(R[{i},j]) = {max_r:.6f}")
        
        db_index = float(np.mean(D_i))
        print(f"\n▶ DAVIES-BOULDIN INDEX:")
        print(f"   DB = (1/{k}) × ({' + '.join([f'{d:.6f}' for d in D_i])})")
        print(f"      = {db_index:.6f}")
    else:
        db_index = float("inf")
        print("\n   ⚠ Chỉ có 1 cụm → Davies-Bouldin = inf")
    
    # =============================================
    # BẢNG TỔNG KẾT
    # =============================================
    print("\n" + "="*100)
    print("BẢNG TỔNG KẾT KẾT QUẢ PHÂN CỤM")
    print("="*100)
    
    print(f"\n{'Cluster':<10} {'Số điểm':<10} {'Centroid':<30} {'Silhouette TB':<15}")
    print("-"*70)
    
    for j in range(k):
        cnt = int(np.sum(labels == j))
        c = centroids[j]
        sil_j = float(np.mean(s_values[labels == j])) if np.any(labels == j) else 0
        print(f"{j:<10} {cnt:<10} ({c[0]:.4f}, {c[1]:.4f}){'':<10} {sil_j:.6f}")
    
    print("-"*70)
    print(f"{'TỔNG':<10} {n_samples:<10} {'':<30} {sil_mean:.6f}")
    print(f"\n📊 Davies-Bouldin Index: {db_index:.6f}")
    print(f"   (DB càng nhỏ → phân cụm càng tốt)")
    
    return labels, centroids, sil_mean, db_index

# ------------------------------------------------------
# Tính MPNN distance và merge core-clusters (Step 5)
# ------------------------------------------------------
def mpnn_distance(clusterA, clusterB, dim):
    """MPNN distance giữa 2 core-cluster, đúng theo mô tả paper.

    - M được tính theo kích thước hai cluster (Equation 1, xấp xỉ):
        M = min( |A|^((T-1)/T), |B|^((T-1)/T) ) với T = dim.
    - Tính khoảng cách giữa TẤT CẢ các cặp điểm của hai cluster.
    - Lấy M khoảng cách nhỏ nhất rồi trả về trung bình đơn giản của chúng.
    """

    # 1. Tính kích thước hai cluster theo số điểm
    size_A = sum(len(g.get("points", [])) for g in clusterA)
    size_B = sum(len(g.get("points", [])) for g in clusterB)

    if size_A == 0 or size_B == 0:
        # Nếu một trong hai cluster không có điểm, coi khoảng cách là vô cùng
        print("[Step 5 - MPNN] Một trong hai cluster không có điểm, trả về inf.")
        return float("inf")

    # 2. Tính M theo công thức xấp xỉ trong paper
    T = dim
    exp_factor = (T - 1) / T if T > 0 else 0.0

    M_A = int(size_A ** exp_factor) if exp_factor > 0 else size_A
    M_B = int(size_B ** exp_factor) if exp_factor > 0 else size_B
    M = min(M_A, M_B)

    if M <= 0:
        M = 1

    # 3. Tính khoảng cách giữa TẤT CẢ các cặp điểm từ 2 cluster
    all_distances = []
    for g1 in clusterA:
        for p1 in g1.get("points", []):
            for g2 in clusterB:
                for p2 in g2.get("points", []):
                    d = euclid(p1, p2)
                    all_distances.append(d)

    if not all_distances:
        # Phòng trường hợp không sinh được khoảng cách nào
        print("[Step 5 - MPNN] Không có cặp điểm nào giữa hai cluster, trả về inf.")
        return float("inf")

    # 4. Sắp xếp và lấy M khoảng cách nhỏ nhất
    all_distances.sort()
    top_M = all_distances[:M]

    # 5. Tính trung bình (weighted average đơn giản)
    mpnn_value = sum(top_M) / len(top_M)
    print(f"[Step 5 - MPNN]   ==> MPNN distance (mean of top-{M}) giữa hai cluster = {mpnn_value:.2f}")
    return mpnn_value


def merge_core_clusters(core_clusters, dim, target_k=None):
    """Step 5: MPNN-based merging of core-clusters (paper-style).

    core_clusters: list các cluster (mỗi cluster là list các core-grid)
    dim: số chiều dữ liệu (T)
    target_k: nếu đặt, lặp merge tới khi len(clusters) <= target_k;
              nếu None → không merge, trả về bản sao.
    """

    # Sao chép shallow list để không sửa trực tiếp đầu vào
    clusters = [c[:] for c in core_clusters]

    # Nếu không yêu cầu giảm số cluster thì giữ nguyên
    if target_k is None:
        return clusters

    # Lặp merge cho tới khi đạt target_k hoặc không merge được nữa
    from itertools import combinations

    while len(clusters) > target_k:
        n = len(clusters)
        best_pair = None
        best_dist = float("inf")

        # Duyệt mọi cặp cluster để tìm cặp có MPNN nhỏ nhất
        for i, j in combinations(range(n), 2):
            d = mpnn_distance(clusters[i], clusters[j], dim)
            if d < best_dist:
                best_dist = d
                best_pair = (i, j)

        if best_pair is None or not math.isfinite(best_dist):
            # Không tìm được cặp hợp lệ để merge (vd. mọi khoảng cách là inf)
            break

        i, j = best_pair
        merged = clusters[i] + clusters[j]

        # Tạo danh sách cluster mới sau khi gộp
        new_clusters = []
        for idx in range(n):
            if idx != i and idx != j:
                new_clusters.append(clusters[idx])
        new_clusters.append(merged)
        clusters = new_clusters

    return clusters


# ------------------------------------------------------
# Vẽ kết quả Step 5: core-clusters sau MPNN (trên lưới đệ quy)
# ------------------------------------------------------
def plot_step5_core_clusters(points, grid_list, merged_clusters,
                             title_prefix="Bước 5: Core-Cluster sau gom MPNN"):
    """Vẽ các core-grid trên lưới đệ quy, tô màu theo cụm sau MPNN.

    - merged_clusters: list các cluster, mỗi cluster là list các grid dict nằm trong grid_list.
    - Non-core grid vẽ xám nhạt, core-grid được tô màu theo id cluster.
    """
    # Tính bounds trực tiếp từ dữ liệu điểm (giống Step 1) để trục đồng nhất
    data = np.asarray(points)
    if len(data) > 0:
        xmin, xmax = float(data[:, 0].min()), float(data[:, 0].max())
        ymin, ymax = float(data[:, 1].min()), float(data[:, 1].max())
    else:
        # fallback từ grid_list nếu không có điểm
        xs_min = [g["min_bin"][0] for g in grid_list]
        xs_max = [g["max_bin"][0] for g in grid_list]
        ys_min = [g["min_bin"][1] for g in grid_list]
        ys_max = [g["max_bin"][1] for g in grid_list]

        xmin, xmax = min(xs_min), max(xs_max)
        ymin, ymax = min(ys_min), max(ys_max)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Map grid id -> màu cụm (chỉ với core-grid)
    color_by_grid_id = {}
    for ci, cluster in enumerate(merged_clusters):
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        for g in cluster:
            color_by_grid_id[id(g)] = color

    # Vẽ toàn bộ grid
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

    # Vẽ toàn bộ điểm dữ liệu (màu xanh dương)
    if len(data) > 0:
        ax.scatter(data[:, 0], data[:, 1], c="blue", s=15, zorder=10, label=f"Points ({len(data)})")

    # Trục dùng đúng bounds từ dữ liệu, không thêm margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("Trục X", fontsize=11)
    ax.set_ylabel("Trục Y", fontsize=11)
    ax.set_title(title_prefix, fontsize=13, fontweight="bold")

    # Legend: từng core-cluster một màu + non-core grid + điểm
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
    # Chừa thêm lề bên phải cho legend để tránh cảnh báo tight_layout
    plt.subplots_adjust(right=0.8)
    plt.show()


# ------------------------------------------------------
# Vẽ kết quả Step 6: lưới + core-grids + điểm tô màu theo cluster
# ------------------------------------------------------
def plot_step5_clusters(points, grid_list, core_clusters, cluster_labels, final_centroids,
                        title_prefix="Kết quả cuối cùng"):
    data = np.asarray(points)
    n_clusters = len(final_centroids)

    # Tính bounds trực tiếp từ dữ liệu điểm (giống Step 1)
    if len(data) > 0:
        xmin, xmax = float(data[:, 0].min()), float(data[:, 0].max())
        ymin, ymax = float(data[:, 1].min()), float(data[:, 1].max())
    else:
        xs_min = [g["min_bin"][0] for g in grid_list]
        xs_max = [g["max_bin"][0] for g in grid_list]
        ys_min = [g["min_bin"][1] for g in grid_list]
        ys_max = [g["max_bin"][1] for g in grid_list]

        xmin, xmax = min(xs_min), max(xs_max)
        ymin, ymax = min(ys_min), max(ys_max)

    # Dùng cùng một khoảng cho cả trục X và Y
    global_min = min(xmin, ymin)
    global_max = max(xmax, ymax)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Màu cho cluster (ô lưới + điểm) dùng bảng màu chung
    cluster_colors = CLUSTER_COLORS

    # Map điểm -> nhãn cluster
    label_by_point = {}
    for p, lab in zip(points, cluster_labels):
        label_by_point[tuple(p)] = int(lab)

    # Vẽ tất cả grid: gán cluster cho ô theo nhãn chiếm đa số của điểm trong ô
    for g in grid_list:
        x0, x1 = g["min_bin"][0], g["max_bin"][0]
        y0, y1 = g["min_bin"][1], g["max_bin"][1]

        labels_in_cell = []
        for p in g.get("points", []):
            lab = label_by_point.get(tuple(p))
            if lab is not None:
                labels_in_cell.append(lab)

        if labels_in_cell:
            # Nhãn chiếm đa số trong ô
            counts = np.bincount(labels_in_cell)
            ci = int(np.argmax(counts))
            facecolor = cluster_colors[ci % len(cluster_colors)]
            alpha = 0.4
            edgecolor = "red"
        else:
            # Ô không chứa điểm nào (hoặc điểm chưa gán cluster) -> xám nhạt
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

    # Vẽ điểm dữ liệu, tô màu theo nhãn cluster
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

    # Cài đặt trục: dùng đúng bounds từ dữ liệu, không thêm margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("Trục X", fontsize=11)
    ax.set_ylabel("Trục Y", fontsize=11)
    ax.set_title(title_prefix, fontsize=13, fontweight="bold")

    # Legend cho các cluster và lưới nền (giống ví dụ)
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
    # Chừa thêm lề bên phải cho legend để tránh cảnh báo tight_layout
    plt.subplots_adjust(right=0.8)
    plt.show()


# ------------------------------------------------------
# Hàm chính STEP 5+6: MPNN merge rồi K-means cho toàn bộ điểm
# ------------------------------------------------------
def step5_cluster_full(points, grid_list, visualize=True, target_k=None):
    """Thực hiện Step 5 và 6:

    - Step 5: Gom core-grids thành các core-cluster (theo adjacency Step 4)
      rồi merge thêm bằng MPNN distance với ngưỡng theta.
    - Step 6: Tính tâm ban đầu từ các core-cluster sau merge và chạy K-means
      cài tay cho toàn bộ điểm (dùng utils.euclid).
    """

    start_time = time.time()

    if not grid_list:
        raise ValueError("grid_list rỗng, không thể chạy Step 5/6")

    dim = len(grid_list[0]["min_bin"])
    print(f"[Step 5] Số ô lưới đầu vào Step 5: {len(grid_list)} (dim={dim})")

    # 1) Gom core-grids thành core-clusters ban đầu (Step 4 kết quả đưa sang)
    core_clusters_raw = build_core_clusters(grid_list, dim)
    print(f"[Step 5] Số core-grid: {sum(g.get('is_core', False) for g in grid_list)}")
    print(f"[Step 5] Số core-cluster ban đầu (từ adjacency): {len(core_clusters_raw)}")

    # 2) MPNN-based merging (Step 5)
    print(f"[Step 5 - MPNN] Bắt đầu merge theo paper-style, target_k={target_k}")
    merged_clusters = merge_core_clusters(core_clusters_raw, dim, target_k=target_k)
    if visualize:
        # Vẽ core-clusters sau MPNN trước khi chạy K-means
        plot_step5_core_clusters(points, grid_list, merged_clusters,
                                 title_prefix="Bước 5: Core-Cluster sau gom MPNN")

    # 3) Tính tâm từng cluster sau merge (làm tâm khởi tạo cho K-means, Step 6)
    init_centroids = compute_initial_centroids(merged_clusters)
    print(f"[Step 6] Số centroid khởi tạo từ core-clusters sau merge: {0 if init_centroids is None else len(init_centroids)}")
    if init_centroids is None or len(init_centroids) == 0:
        print("\n[Step 6] Không có centroid core-cluster nào sau MPNN merge.")
        raise ValueError("Step 6: Không thể chạy K-means vì không có core-cluster centroid khởi tạo.")

    # 4) Step 6: K-means cho toàn bộ điểm (cài tay, dùng euclid)
    data = np.asarray(points)
    print(f"[Step 6] Bắt đầu K-means cho toàn bộ {len(data)} điểm.")
    cluster_labels, final_centroids = kmeans_assign_all_points_custom(data, init_centroids)
    print("[Step 6] Hoàn thành K-means.")
    print("[Step 6] KẾT QUẢ THEO TỪNG CLUSTER:")
    for ci, c in enumerate(final_centroids):
        cnt = int(np.sum(cluster_labels == ci))
        print(f"  - Cluster {ci}: số điểm = {cnt}, centroid = {c}")

    # 4b) Tính và in chỉ số Silhouette
    sil_scores, sil_mean = compute_silhouette(points, cluster_labels)
    print("\n[Step 6] CHỈ SỐ SILHOUETTE:")
    print(f"  - Silhouette trung bình cho toàn bộ phân cụm = {sil_mean:.2f}")

    # Silhouette trung bình theo từng cluster
    labels_arr = np.asarray(cluster_labels)
    n_clusters = len(final_centroids)
    cluster_silhouette_stats = []
    for ci in range(n_clusters):
        mask = labels_arr == ci
        if not np.any(mask):
            continue
        sil_ci = float(sil_scores[mask].mean())
        count_ci = int(np.sum(mask))
        cluster_silhouette_stats.append((ci, sil_ci, count_ci))
        print(f"  - Cluster {ci}: Silhouette trung bình = {sil_ci:.2f} (số điểm = {count_ci})")

    # Một vài điểm "khó phân cụm" (Silhouette thấp nhất)
    k = min(10, len(points))
    hard_points_info = []
    if k > 0:
        print("\n  - Top các điểm có Silhouette thấp nhất (khó phân cụm):")
        idx_sorted = np.argsort(sil_scores)  # tăng dần
        for rank in range(k):
            i = int(idx_sorted[rank])
            p = points[i]
            ci = int(labels_arr[i])
            s_i = float(sil_scores[i])
            hard_points_info.append((i, p, ci, s_i))
            print(f"      + Điểm {i}: p={p}, cluster={ci}, silhouette={s_i:.2f}")

    # Ghi kết quả Silhouette ra file TXT
    # --- Compute Davies-Bouldin index for the clustering ---
    labels_arr = np.asarray(cluster_labels)
    db_index = compute_davies_bouldin(data, labels_arr, final_centroids)
    print(f"\n[Step 6] Davies-Bouldin index (toàn hệ thống) = {db_index:.6f}")

    try:
        with open("silhouette_results_demo.txt", "w", encoding="utf-8") as f:
            f.write("[Step 6] CHỈ SỐ SILHOUETTE\n")
            f.write(f"Silhouette trung bình toàn bộ: {sil_mean:.6f}\n\n")

            f.write("Silhouette trung bình theo từng cluster:\n")
            for ci, sil_ci, count_ci in cluster_silhouette_stats:
                f.write(f"- Cluster {ci}: silhouette_mean={sil_ci:.6f}, so_diem={count_ci}\n")

            f.write("\nTop các điểm có Silhouette thấp nhất (khó phân cụm):\n")
            for i, p, ci, s_i in hard_points_info:
                f.write(
                    f"+ Diem {i}: x={float(p[0]):.6f}, y={float(p[1]):.6f}, "
                    f"cluster={ci}, silhouette={s_i:.6f}\n"
                )
            # Write Davies-Bouldin index
            f.write(f"\nDavies-Bouldin = {db_index:.6f}\n")
        print("\n[Step 6] Đã ghi kết quả Silhouette ra file silhouette_results_demo.txt")
    except Exception as e:
        print(f"\n[Step 6] Lỗi khi ghi file silhouette_results_caitien_demo.txt: {e}")

    # 5) Vẽ kết quả cuối cùng nếu cần (dựa trên format Step 2/3)
    if visualize:
        plot_step5_clusters(points, grid_list, merged_clusters, cluster_labels, final_centroids,
                            title_prefix="Bước 6: Kết quả phân cụm cuối cùng")

    total_time = time.time() - start_time
    print(f"\n[Timing] Thời gian xử lý Step 5+6: {total_time:.4f} giây")

    return cluster_labels, merged_clusters, final_centroids, sil_mean, db_index, total_time


if __name__ == "__main__":
    """Demo đầy đủ pipeline: Step 1 -> 2 -> 3 -> 4 -> 5.

    Chỉ dùng khi chạy trực tiếp file step5_core_clustering.py.
    """
    
    # ========= CẤU HÌNH =========
    # Đặt True để hiển thị đồ thị, False để tắt (tránh chờ đóng cửa sổ)
    SHOW_PLOTS = True
    # ============================

    from step1_compute_M_R import step1_compute_original
    from utils import load_data_txt
    from step3_recursive_partitioning import step3_handle_dense_grids
    from step2_grid_classification import build_grid, classify_grids, plot_classification

    data_path = "data.txt"

    # Đo thời gian cho toàn bộ hệ thống (pipeline Step 1 -> 6)
    total_start = time.time()

    # 1) Step 1: tìm M, R
    print("===== STEP 1: Tính M, R =====")
    step1_result = step1_compute_original(data_path, K=10, max_M=200)
    M = step1_result["M"]
    R = step1_result["R"]
    print(f"[Step 1] M = {M}, R = {R}")

    # 2) Đọc dữ liệu
    print("\n===== ĐỌC DỮ LIỆU =====")
    points = load_data_txt(data_path)
    print(f"[Data] Số điểm đọc được: {len(points)}")

    # 2) Step 2: lưới tĩnh + phân loại
    print("\n===== STEP 2: Xây lưới tĩnh và phân loại =====")
    grid_step2, bounds = build_grid(points, M)
    print(f"[Step 2] Số ô lưới tĩnh: {len(grid_step2)}")
    classified_step2 = classify_grids(grid_step2, R)
    for gtype in ["core", "dense", "sparse", "empty"]:
        cells = classified_step2.get(gtype, [])
        print(f"[Step 2] Số ô loại {gtype}: {len(cells)}")
    if SHOW_PLOTS:
        plot_classification(points, grid_step2, classified_step2, bounds, M, R)

    # 3) Step 3: xây lưới đệ quy và phân loại core/dense/sparse/empty
    print("\n===== STEP 3: Lưới đệ quy và phân loại =====")
    step3_result = step3_handle_dense_grids(points, M, R, bounds, visualize=SHOW_PLOTS)
    final_cells = step3_result["final_cells"]
    print(f"[Step 3] Số ô cuối cùng (final_cells): {len(final_cells)}")

    # 4) Step 4: Gom core-grids trên lưới tĩnh Step 2 và vẽ
    print("\n===== STEP 4: Gom core-grids trên lưới tĩnh Step 2 =====")
    grid_list_step4 = []
    for (ix, iy), cell in grid_step2.items():
        # Lưu ý: sau refactor, ô lưới Step 2 là GridCell, không còn là dict.
        # Lấy loại ô từ thuộc tính grid_type, nếu chưa có thì coi là 'unclassified'.
        gtype = getattr(cell, "grid_type", "unclassified")
        is_core = gtype == "core" or cell in classified_step2.get("core", [])

        # Chuyển về dict với min_bin/max_bin/points/is_core giống format Step 4/5 dùng chung.
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
    if SHOW_PLOTS:
        plot_core_groups(points, grid_list_step4, core_clusters_step4,
                         title_prefix="Bước 4: Gom Core Grid thành Cluster")

    # 5) Step 5+6: Gom core-grids từ lưới đệ quy Step 3 + MPNN + K-means toàn bộ điểm
    print("\n===== STEP 5+6: MPNN merge + K-means toàn bộ điểm =====")
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
        points, grid_list_step5, visualize=SHOW_PLOTS
    )

    print("\n===== KẾT QUẢ STEP 6 (sau MPNN + K-means) =====")
    print(f"Số cluster (từ core-clusters sau merge): {len(core_clusters)}")
    print(f"Số điểm: {len(points)}")
    print(f"Silhouette trung bình toàn bộ: {sil_mean:.4f}")
    print(f"Davies-Bouldin index toàn hệ thống: {db_index:.6f}")
    print(f"Thời gian Step 5+6 (bên trong hàm): {step56_time:.4f} giây")
    print("Centroids cuối cùng:")
    for idx, c in enumerate(final_centroids):
        print(f"  Cluster {idx}: centroid = {c}")

    total_end = time.time()
    total_runtime = total_end - total_start
    print(f"\n[Timing] Thời gian ch ạy TOÀN BỘ HỆ THỐNG (Step 1 -> 6): {total_runtime:.4f} giây")

    # Ghi ra file TXT chỉ chứa 2 chỉ số: Time(s) toàn bộ hệ thống và Silhouette trung bình
    try:
        with open("time_silhouette_results_demo.txt", "w", encoding="utf-8") as f:
            f.write("Time(s)_full_system = {:.6f}\n".format(total_runtime))
            f.write("Silhouette_mean = {:.6f}\n".format(sil_mean))
            f.write("Davies_Bouldin = {:.6f}\n".format(db_index))
        print("[Output] Đã ghi 2 chỉ số Time(s) và Silhouette vào file time_silhouette_result_demo.txt")
    except Exception as e:
        print(f"[Output] Lỗi khi ghi file time_silhouette_result_demo.txt: {e}")
