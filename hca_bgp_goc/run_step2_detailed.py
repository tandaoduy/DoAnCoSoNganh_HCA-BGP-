"""Script để chạy Step 2 và lưu output chi tiết"""
from step1_compute_M_R import step1_compute_original
from step2_grid_classification import step2_classify_grids
from utils import load_data_txt
import sys

# Redirect output to file
original_stdout = sys.stdout
with open('output_step2.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f
    
    data_path = 'data.txt'
    step1_result = step1_compute_original(data_path, K=3)
    M = step1_result['M']
    R = step1_result['R']
    points = load_data_txt(data_path)
    step2_result = step2_classify_grids(points, M, R, visualize=False, show_detailed=True)
    
    sys.stdout = original_stdout

print("Output đã được lưu vào output_step2.txt")
