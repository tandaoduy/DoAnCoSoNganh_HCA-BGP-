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
    """
    Algorithm 1: denseUnitsToClusters (theo bài báo)
    
    Input: denseUnitsND (core_grids), dataset
    Output: clusters - danh sách các core-cluster
    """
    core_grids = [g for g in grid_list if g["is_core"]]
    L = len(core_grids)  # Line 1: L ← len(denseUnitsND)
    
    if L == 0:
        return []
    
    # Line 2: initialization C[0:L], C[] ← -1
    C = [-1] * L  # Cluster label cho mỗi dense grid
    
    # Line 3: initialization EK ← -1
    EK = -1  # Cluster ID hiện tại
    
    # Line 5: for i:0 to L
    for i in range(L):
        # Line 6: if C[i] == -1 then
        if C[i] == -1:
            # Line 7: EK ← EK+1, C[i] ← EK
            EK = EK + 1
            C[i] = EK
            
            # Line 8: initialization Current []
            Current = []
            
            # Line 9-12: for j:0 to L, tìm các grid kề với grid i
            for j in range(L):
                if C[j] == -1:
                    # Line 10: if C[j] == -1 and distance(i,j) <= 1
                    d = grid_distance(core_grids[i], core_grids[j], dim)
                    if d <= 1.0:  # adjacent if distance <= 1 (Algorithm 1, line 10)
                        # Line 11: C[j] ← EK
                        C[j] = EK
                        # Line 12: add C[j] to Current
                        Current.append(j)
            
            # Line 13-17: for y in Current, lan truyền tìm thêm grid kề
            idx = 0
            while idx < len(Current):
                y = Current[idx]
                # Line 14: for s:0 to L
                for s in range(L):
                    # Line 15: if C[s] == -1 and distance(y,s) <= 1
                    if C[s] == -1:
                        d = grid_distance(core_grids[y], core_grids[s], dim)
                        if d <= 1.0:
                            # Line 16: C[s] ← EK
                            C[s] = EK
                            # Line 17: add C[s] to Current
                            Current.append(s)
                idx += 1
    
    # Tạo danh sách clusters từ C
    clusters = []
    for cluster_id in range(EK + 1):
        cluster = [core_grids[i] for i in range(L) if C[i] == cluster_id]
        if cluster:
            clusters.append(cluster)
    
    return clusters


# =============================
# IN CHI TIẾT CÔNG THỨC TÍNH KHOẢNG CÁCH KỀ (STEP 4)
# =============================
def print_adjacency_formulas_detail(grid_list, dim=2):
    """
    In chi tiết công thức tính khoảng cách kề giữa các core-grid
    theo Definition 9 và Definition 10 trong paper.
    """
    print("\n" + "="*100)
    print("CHI TIẾT CÔNG THỨC TÍNH KHOẢNG CÁCH KỀ GIỮA CÁC CORE-GRID (STEP 4)")
    print("="*100)
    
    # =============================================
    # PHẦN 1: CÔNG THỨC ĐỊNH NGHĨA
    # =============================================
    print("\n" + "─"*100)
    print("📐 DEFINITION 9: KHOẢNG CÁCH 1 CHIỀU (Axis Distance)")
    print("─"*100)
    print("\n▶ CÔNG THỨC:")
    print("   distance_i(a, b) = min(|maxA - maxB|, |minA - minB|) / min(lenA, lenB)")
    print("\n   Trong đó:")
    print("   • a, b: hai ô lưới (grid)")
    print("   • maxA, minA: biên lớn nhất/nhỏ nhất của ô a theo chiều i")
    print("   • maxB, minB: biên lớn nhất/nhỏ nhất của ô b theo chiều i")
    print("   • lenA = maxA - minA (độ rộng ô a)")
    print("   • lenB = maxB - minB (độ rộng ô b)")
    
    print("\n" + "─"*100)
    print("📐 DEFINITION 10: KHOẢNG CÁCH N CHIỀU (Grid Distance)")
    print("─"*100)
    print("\n▶ CÔNG THỨC:")
    print("   distance(a, b)_N = Σᵢ distance_i(a, b)")
    print("\n   → Tổng khoảng cách theo từng chiều (với dữ liệu 2D: i = x, y)")
    
    print("\n" + "─"*100)
    print("📐 ĐIỀU KIỆN KỀ NHAU (Adjacency Condition)")
    print("─"*100)
    print("\n▶ HAI Ô ĐƯỢC COI LÀ KỀ NHAU NẾU:")
    print("   • Lưới tĩnh (có ix, iy): |ix₁ - ix₂| ≤ 1 VÀ |iy₁ - iy₂| ≤ 1")
    print("   • Lưới đệ quy (không có ix, iy): distance(a, b)_N < 1.0")
    
    # =============================================
    # PHẦN 2: LỌC CORE GRIDS
    # =============================================
    core_grids = [g for g in grid_list if g.get("is_core", False)]
    N = len(core_grids)
    
    print("\n" + "─"*100)
    print("📊 DANH SÁCH CORE GRIDS")
    print("─"*100)
    print(f"\n▶ Tổng số grid: {len(grid_list)}")
    print(f"▶ Số Core Grid: {N}")
    
    if N == 0:
        print("\n   ⚠ Không có Core Grid nào để tính khoảng cách!")
        return []
    
    print(f"\n{'#':<4} {'(ix,iy)':<12} {'X range':<25} {'Y range':<25} {'Điểm':<6}")
    print("-"*80)
    for i, g in enumerate(core_grids):
        ix = g.get("ix", "-")
        iy = g.get("iy", "-")
        x_range = f"[{g['min_bin'][0]:.4f}, {g['max_bin'][0]:.4f})"
        y_range = f"[{g['min_bin'][1]:.4f}, {g['max_bin'][1]:.4f})"
        n_pts = len(g.get("points", []))
        print(f"{i:<4} ({ix},{iy}){'':<5} {x_range:<25} {y_range:<25} {n_pts:<6}")
    
    # =============================================
    # PHẦN 3: TÍNH KHOẢNG CÁCH CHI TIẾT
    # =============================================
    print("\n" + "─"*100)
    print("📊 TÍNH KHOẢNG CÁCH GIỮA CÁC CẶP CORE GRID")
    print("─"*100)
    
    if N < 2:
        print("\n   ⚠ Chỉ có 1 Core Grid, không tính khoảng cách cặp!")
        return [[core_grids[0]]]
    
    # Ma trận kề
    adj = [[0] * N for _ in range(N)]
    
    print(f"\n▶ TÍNH CHI TIẾT CHO TỪNG CẶP:")
    
    pair_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            g1 = core_grids[i]
            g2 = core_grids[j]
            pair_count += 1
            
            # Lấy thông tin
            ix1, iy1 = g1.get("ix", None), g1.get("iy", None)
            ix2, iy2 = g2.get("ix", None), g2.get("iy", None)
            
            print(f"\n   ┌───────────────────────────────────────────────────────────────────────────────┐")
            print(f"   │ CẶP {pair_count}: Grid {i} - Grid {j}")
            print(f"   └───────────────────────────────────────────────────────────────────────────────┘")
            
            # Grid 1 info
            print(f"\n   Grid {i}:")
            if ix1 is not None:
                print(f"   • (ix, iy) = ({ix1}, {iy1})")
            print(f"   • X: [{g1['min_bin'][0]:.4f}, {g1['max_bin'][0]:.4f}]")
            print(f"   • Y: [{g1['min_bin'][1]:.4f}, {g1['max_bin'][1]:.4f}]")
            
            # Grid 2 info
            print(f"\n   Grid {j}:")
            if ix2 is not None:
                print(f"   • (ix, iy) = ({ix2}, {iy2})")
            print(f"   • X: [{g2['min_bin'][0]:.4f}, {g2['max_bin'][0]:.4f}]")
            print(f"   • Y: [{g2['min_bin'][1]:.4f}, {g2['max_bin'][1]:.4f}]")
            
            # Kiểm tra phương pháp
            if ix1 is not None and iy1 is not None and ix2 is not None and iy2 is not None:
                # Phương pháp 1: Lưới tĩnh
                print(f"\n   📌 PHƯƠNG PHÁP: Kiểm tra chỉ số (ix, iy)")
                diff_ix = abs(ix1 - ix2)
                diff_iy = abs(iy1 - iy2)
                print(f"   • |ix₁ - ix₂| = |{ix1} - {ix2}| = {diff_ix}")
                print(f"   • |iy₁ - iy₂| = |{iy1} - {iy2}| = {diff_iy}")
                print(f"   • Điều kiện kề: |Δix| ≤ 1 VÀ |Δiy| ≤ 1")
                
                is_adj = (diff_ix <= 1 and diff_iy <= 1)
                if is_adj:
                    adj[i][j] = adj[j][i] = 1
                    print(f"\n   ═══════════════════════════════════════════════════════════════")
                    print(f"   ✅ KẾT LUẬN: KỀ NHAU ({diff_ix} ≤ 1 VÀ {diff_iy} ≤ 1)")
                    print(f"   ═══════════════════════════════════════════════════════════════")
                else:
                    print(f"\n   ═══════════════════════════════════════════════════════════════")
                    print(f"   ❌ KẾT LUẬN: KHÔNG KỀ NHAU")
                    print(f"   ═══════════════════════════════════════════════════════════════")
            else:
                # Phương pháp 2: Lưới đệ quy - dùng Definition 9, 10
                print(f"\n   📌 PHƯƠNG PHÁP: Tính khoảng cách theo Definition 9, 10")
                
                # Tính distance_x (chiều X)
                a_min_x, a_max_x = g1['min_bin'][0], g1['max_bin'][0]
                b_min_x, b_max_x = g2['min_bin'][0], g2['max_bin'][0]
                len_a_x = abs(a_max_x - a_min_x)
                len_b_x = abs(b_max_x - b_min_x)
                
                print(f"\n   ▶ distance_x (Definition 9 - chiều X):")
                print(f"      Grid {i}: minX = {a_min_x:.4f}, maxX = {a_max_x:.4f}, lenX = {len_a_x:.4f}")
                print(f"      Grid {j}: minX = {b_min_x:.4f}, maxX = {b_max_x:.4f}, lenX = {len_b_x:.4f}")
                
                denom_x = min(len_a_x, len_b_x)
                if denom_x > 0:
                    diff_max_x = abs(a_max_x - b_max_x)
                    diff_min_x = abs(a_min_x - b_min_x)
                    num_x = min(diff_max_x, diff_min_x)
                    d_x = num_x / denom_x
                    print(f"      |maxA - maxB| = |{a_max_x:.4f} - {b_max_x:.4f}| = {diff_max_x:.4f}")
                    print(f"      |minA - minB| = |{a_min_x:.4f} - {b_min_x:.4f}| = {diff_min_x:.4f}")
                    print(f"      min(lenA, lenB) = min({len_a_x:.4f}, {len_b_x:.4f}) = {denom_x:.4f}")
                    print(f"      distance_x = min({diff_max_x:.4f}, {diff_min_x:.4f}) / {denom_x:.4f}")
                    print(f"                = {num_x:.4f} / {denom_x:.4f} = {d_x:.6f}")
                else:
                    d_x = 0.0
                    print(f"      min(lenA, lenB) = 0 → distance_x = 0")
                
                # Tính distance_y (chiều Y)
                a_min_y, a_max_y = g1['min_bin'][1], g1['max_bin'][1]
                b_min_y, b_max_y = g2['min_bin'][1], g2['max_bin'][1]
                len_a_y = abs(a_max_y - a_min_y)
                len_b_y = abs(b_max_y - b_min_y)
                
                print(f"\n   ▶ distance_y (Definition 9 - chiều Y):")
                print(f"      Grid {i}: minY = {a_min_y:.4f}, maxY = {a_max_y:.4f}, lenY = {len_a_y:.4f}")
                print(f"      Grid {j}: minY = {b_min_y:.4f}, maxY = {b_max_y:.4f}, lenY = {len_b_y:.4f}")
                
                denom_y = min(len_a_y, len_b_y)
                if denom_y > 0:
                    diff_max_y = abs(a_max_y - b_max_y)
                    diff_min_y = abs(a_min_y - b_min_y)
                    num_y = min(diff_max_y, diff_min_y)
                    d_y = num_y / denom_y
                    print(f"      |maxA - maxB| = |{a_max_y:.4f} - {b_max_y:.4f}| = {diff_max_y:.4f}")
                    print(f"      |minA - minB| = |{a_min_y:.4f} - {b_min_y:.4f}| = {diff_min_y:.4f}")
                    print(f"      min(lenA, lenB) = min({len_a_y:.4f}, {len_b_y:.4f}) = {denom_y:.4f}")
                    print(f"      distance_y = min({diff_max_y:.4f}, {diff_min_y:.4f}) / {denom_y:.4f}")
                    print(f"                = {num_y:.4f} / {denom_y:.4f} = {d_y:.6f}")
                else:
                    d_y = 0.0
                    print(f"      min(lenA, lenB) = 0 → distance_y = 0")
                
                # Tổng khoảng cách (Definition 10)
                total_d = d_x + d_y
                print(f"\n   ▶ distance_2D (Definition 10):")
                print(f"      distance = distance_x + distance_y")
                print(f"               = {d_x:.6f} + {d_y:.6f}")
                print(f"               = {total_d:.6f}")
                
                is_adj = (total_d < 1.0)
                if is_adj:
                    adj[i][j] = adj[j][i] = 1
                    print(f"\n   ═══════════════════════════════════════════════════════════════")
                    print(f"   ✅ KẾT LUẬN: KỀ NHAU (distance = {total_d:.6f} < 1.0)")
                    print(f"   ═══════════════════════════════════════════════════════════════")
                else:
                    print(f"\n   ═══════════════════════════════════════════════════════════════")
                    print(f"   ❌ KẾT LUẬN: KHÔNG KỀ NHAU (distance = {total_d:.6f} >= 1.0)")
                    print(f"   ═══════════════════════════════════════════════════════════════")
    
    # =============================================
    # PHẦN 4: MA TRẬN KỀ
    # =============================================
    print("\n" + "─"*100)
    print("📊 MA TRẬN KỀ (Adjacency Matrix)")
    print("─"*100)
    
    print("\n      ", end="")
    for j in range(N):
        print(f"G{j:<3}", end="")
    print()
    
    for i in range(N):
        print(f"   G{i} ", end="")
        for j in range(N):
            if i == j:
                print("  - ", end="")
            else:
                print(f"  {adj[i][j]} ", end="")
        print()
    
    # =============================================
    # PHẦN 5: GOM CLUSTER THEO ALGORITHM 1 (denseUnitsToClusters)
    # =============================================
    print("\n" + "─"*100)
    print("📊 GOM CÁC CORE GRID THÀNH CORE-CLUSTERS (Algorithm 1)")
    print("─"*100)
    
    print("\n▶ ALGORITHM 1: denseUnitsToClusters")
    print("   Input: denseUnitsND (core grids), dataset")
    print("   Output: clusters - danh sách các core-cluster")
    
    # Line 1: L ← len(denseUnitsND)
    L = N
    print(f"\n   Line 1: L ← {L}")
    
    # Line 2: initialization C[0:L], C[] ← -1
    C = [-1] * L
    print(f"   Line 2: C[] ← [-1] * {L}")
    
    # Line 3: initialization EK ← -1
    EK = -1
    print(f"   Line 3: EK ← -1")
    
    print(f"\n▶ TIẾN HÀNH GOM CÁC CORE GRID:")
    
    # Line 5: for i:0 to L
    for i in range(L):
        # Line 6: if C[i] == -1 then
        if C[i] == -1:
            # Line 7: EK ← EK+1, C[i] ← EK
            EK = EK + 1
            C[i] = EK
            print(f"\n   🔹 Grid {i} chưa có cluster → Tạo cluster mới EK={EK}")
            
            # Line 8: initialization Current []
            Current = []
            
            # Line 9-12: tìm các grid kề với grid i
            for j in range(L):
                if C[j] == -1:
                    d = grid_distance(core_grids[i], core_grids[j], dim)
                    if d <= 1.0:
                        C[j] = EK
                        Current.append(j)
                        print(f"      → Grid {j} kề với Grid {i} (distance={d:.4f} <= 1) → C[{j}]={EK}")
            
            # Line 13-17: lan truyền tìm thêm grid kề
            idx = 0
            while idx < len(Current):
                y = Current[idx]
                for s in range(L):
                    if C[s] == -1:
                        d = grid_distance(core_grids[y], core_grids[s], dim)
                        if d <= 1.0:
                            C[s] = EK
                            Current.append(s)
                            print(f"      → Grid {s} kề với Grid {y} (distance={d:.4f} <= 1) → C[{s}]={EK}")
                idx += 1
    
    # Tạo danh sách clusters từ C
    clusters = []
    for cluster_id in range(EK + 1):
        cluster_members = [i for i in range(L) if C[i] == cluster_id]
        clusters.append([core_grids[i] for i in cluster_members])
    
    print(f"\n▶ KẾT QUẢ:")
    print(f"   C[] = {C}")
    for ci, cluster in enumerate(clusters):
        members = [i for i in range(L) if C[i] == ci]
        print(f"   → Cluster {ci}: gồm các Grid {members}")
    
    # =============================================
    # PHẦN 6: BẢNG TỔNG KẾT
    # =============================================
    print("\n" + "="*100)
    print("BẢNG TỔNG KẾT CORE-CLUSTERS")
    print("="*100)
    
    print(f"\n▶ Tổng số Core Grid: {N}")
    print(f"▶ Tổng số Core-Cluster: {len(clusters)}")
    
    print(f"\n{'Cluster':<10} {'Số Grid':<10} {'Danh sách Grid':<40} {'Tổng điểm':<10}")
    print("-"*75)
    
    for ci, cluster in enumerate(clusters):
        grid_ids = []
        total_pts = 0
        for g in cluster:
            idx = core_grids.index(g)
            grid_ids.append(idx)
            total_pts += len(g.get("points", []))
        
        grid_str = ", ".join([f"G{idx}" for idx in grid_ids])
        print(f"{ci:<10} {len(cluster):<10} {grid_str:<40} {total_pts:<10}")
    
    print("-"*75)
    
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
def plot_core_groups(points, grid_list, core_clusters, title_prefix="Bước 4: Gom Core Grid thành Cluster"):
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
    ax.set_title(f'Bước 4: Lọc tất cả các Core Dense Grid',
                 fontsize=13, fontweight='bold')

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
                     title_prefix="Bước 4: Gom Core Grid thành Cluster")
