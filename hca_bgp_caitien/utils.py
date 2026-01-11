"""Utility helpers used across HCA-BGP pipeline.

Includes simple distance functions and I/O helpers.
"""

import math

def euclid(p1, p2):
    """Tính khoảng cách Euclid 2 chiều."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def load_data_txt(path):
    """
    Đọc file dữ liệu .txt có dạng:
    PoiID   NEAR_X  NEAR_Y

    - Hỗ trợ file nhỏ và file rất lớn.
    - Tự bỏ qua dòng trống, dòng lỗi.
    - Trả về list các điểm: [(x, y), ...]
    """
    points = []

    with open(path, 'r', encoding='utf-8') as f:
        first = True
        for line in f:
            line = line.strip()

            if not line:
                continue   # bỏ dòng trống

            # bỏ dòng header
            if first:
                first = False
                continue

            parts = line.split()
            if len(parts) < 3:
                continue   # dòng lỗi

            try:
                x = float(parts[1])
                y = float(parts[2])
                points.append((x, y))
            except:
                # nếu có số bị lỗi thì bỏ qua
                continue

    return points
