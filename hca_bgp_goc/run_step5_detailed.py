"""Script để chạy Step 5+6 và in chi tiết công thức K-means, Euclidean, Silhouette, Davies-Bouldin"""
from step1_compute_M_R import step1_compute_original
from step2_grid_classification import build_grid, classify_grids
from step3_recursive_partitioning import step3_handle_dense_grids
from step4_core_grouping import build_core_clusters, compute_initial_centroids
from step5_core_clustering import print_kmeans_formulas_detail
from utils import load_data_txt
import sys

# Redirect output to file
original_stdout = sys.stdout
with open('output_step5_kmeans.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f
    
    data_path = 'data.txt'
    
    # Step 1
    print("===== STEP 1: Tính M, R =====")
    step1_result = step1_compute_original(data_path, K=3)
    M = step1_result['M']
    R = step1_result['R']
    print(f"M = {M}, R = {R}")
    
    # Load data
    points = load_data_txt(data_path)
    
    # Step 2
    grid, bounds = build_grid(points, M)
    
    # Step 3
    step3_result = step3_handle_dense_grids(points, M, R, bounds, visualize=False, show_detailed=False)
    final_cells = step3_result['final_cells']
    
    # Convert to grid_list format
    grid_list = []
    for cell in final_cells:
        is_core = getattr(cell, 'grid_type', None) == 'core'
        grid_list.append({
            'min_bin': (cell.xmin, cell.ymin),
            'max_bin': (cell.xmax, cell.ymax),
            'points': list(getattr(cell, 'points', [])),
            'is_core': is_core,
        })
    
    # Step 4: Build core clusters
    dim = 2
    core_clusters = build_core_clusters(grid_list, dim)
    init_centroids = compute_initial_centroids(core_clusters)
    
    print(f"\n===== STEP 4: Core Clusters =====")
    print(f"Số core-clusters: {len(core_clusters)}")
    print(f"Tâm cụm ban đầu:")
    for i, c in enumerate(init_centroids):
        print(f"  Cluster {i}: ({c[0]:.4f}, {c[1]:.4f})")
    
    # In chi tiết K-means, Euclidean, Silhouette, Davies-Bouldin
    print_kmeans_formulas_detail(points, init_centroids)
    
    sys.stdout = original_stdout

print("Output đã được lưu vào output_step5_kmeans.txt")
