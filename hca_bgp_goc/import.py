import random

def generate_large_data(filename="data.txt", n=100, xmin=1, xmax=10):
    with open(filename, "w") as f:
        f.write("PoiID\tNEAR_X\tNEAR_Y\n")
        for i in range(1, n+1):
            x = round(random.uniform(xmin, xmax), 3)
            y = round(random.uniform(xmin, xmax), 3)
            f.write(f"{i}\t{x}\t{y}\n")

# Gọi hàm để tạo file với 1000 điểm
generate_large_data("data.txt", n =20)
