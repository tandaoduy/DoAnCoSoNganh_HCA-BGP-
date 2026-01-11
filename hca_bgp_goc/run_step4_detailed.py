"""Script để chạy Step 4 và in chi tiết công thức khoảng cách kề"""
from step1_compute_M_R import step1_compute_original
from step2_grid_classification import build_grid, classify_grids
from step3_recursive_partitioning import step3_handle_dense_grids
from step4_core_grouping import print_adjacency_formulas_detail
from utils import load_data_txt
import sys

# Redirect output to file
original_stdout = sys.stdout
with open('output_step4_adjacency.txt', 'w', encoding='utf-8') as f:
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
        # Thêm ix, iy nếu có
        ix = getattr(cell, 'ix', None)
        iy = getattr(cell, 'iy', None)
        
        entry = {
            'min_bin': (cell.xmin, cell.ymin),
            'max_bin': (cell.xmax, cell.ymax),
            'points': list(getattr(cell, 'points', [])),
            'is_core': is_core,
        }
        if ix is not None and iy is not None:
            entry['ix'] = ix
            entry['iy'] = iy
        
        grid_list.append(entry)
    
    # In chi tiết Step 4
    print_adjacency_formulas_detail(grid_list, dim=2)
    
    sys.stdout = original_stdout

print("Output đã được lưu vào output_step4_adjacency.txt")
