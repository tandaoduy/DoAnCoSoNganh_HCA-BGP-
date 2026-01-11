"""Script để chạy Step 3 và lưu output chi tiết chia lưới đệ quy"""
from step1_compute_M_R import step1_compute_original
from step2_grid_classification import build_grid, classify_grids
from step3_recursive_partitioning import step3_handle_dense_grids
from utils import load_data_txt
import sys

# Redirect output to file
original_stdout = sys.stdout
with open('output_step3.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f
    
    data_path = 'data.txt'
    
    # Step 1
    step1_result = step1_compute_original(data_path, K=3)
    M = step1_result['M']
    R = step1_result['R']
    points = load_data_txt(data_path)
    
    # Step 2 (chỉ để lấy bounds)
    grid, bounds = build_grid(points, M)
    
    # Step 3 với show_detailed=True
    step3_result = step3_handle_dense_grids(points, M, R, bounds, visualize=False, show_detailed=True)
    
    sys.stdout = original_stdout

print("Output đã được lưu vào output_step3.txt")
