import math


def diagonal_length(xmin, xmax, ymin, ymax):
    """Chiều dài đường chéo của một hình chữ nhật (ô lưới)."""
    dx = xmax - xmin
    dy = ymax - ymin
    return math.sqrt(dx * dx + dy * dy)


def compute_centroid(points):
    """Trọng tâm các điểm (Cj). Trả về None nếu không có điểm."""
    if not points:
        return None
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    n = len(points)
    return sx / n, sy / n


def compute_pj_for_cell(points, xmin, xmax, ymin, ymax):
    """Tính pj cho một "ô" xác định bởi danh sách điểm + bounding box.

    pj = ||Gj - Cj|| / diagonal
    Trong đó Gj là tâm hình học của box, Cj là trọng tâm các điểm.
    """
    Cj = compute_centroid(points)
    if Cj is None:
        return None

    # Tâm hình học Gj
    Gjx = (xmin + xmax) / 2.0
    Gjy = (ymin + ymax) / 2.0

    # Khoảng cách giữa Gj và Cj (tử số)
    num = math.sqrt((Gjx - Cj[0])**2 + (Gjy - Cj[1])**2)

    # Đường chéo (mẫu số)
    denom = diagonal_length(xmin, xmax, ymin, ymax)
    if denom == 0:
        return 0.0

    return num / denom


def compute_Dj_for_cell(points, xmin, xmax, ymin, ymax):
    """Tính Dj cho một "ô" xác định bởi danh sách điểm + bounding box.

    Dj = std_distance / diagonal
    std_distance được xấp xỉ bằng căn bậc hai của trung bình bình phương khoảng
    cách từ điểm tới trọng tâm Cj.
    """
    Cj = compute_centroid(points)
    if Cj is None:
        return None

    if not points:
        return 0.0

    distances_squared = [
        (p[0] - Cj[0])**2 + (p[1] - Cj[1])**2 for p in points
    ]
    if not distances_squared:
        return 0.0

    mean_dist = math.sqrt(sum(distances_squared) / len(distances_squared))
    denom = diagonal_length(xmin, xmax, ymin, ymax)
    if denom == 0:
        return 0.0

    return mean_dist / denom
