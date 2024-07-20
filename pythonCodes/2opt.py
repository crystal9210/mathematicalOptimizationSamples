import itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
import time

# 日本語フォントの設定（Macの標準日本語フォントを使用）
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['font.sans-serif'] = ['Hiragino Sans']

# 距離行列の作成
def create_distance_matrix(coords: List[Tuple[int, int]]) -> np.ndarray:
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
    return distance_matrix

# 巡回路の距離を計算
def calculate_total_distance(path: List[int], distance_matrix: np.ndarray) -> float:
    return sum(distance_matrix[path[i], path[(i + 1) % len(path)]] for i in range(len(path)))

# 2-optによる巡回路の改善
def two_opt(path: List[int], distance_matrix: np.ndarray) -> Tuple[float, List[int]]:
    best_distance = calculate_total_distance(path, distance_matrix)
    best_path = path.copy()
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path)):
                if j - i == 1: continue  # 連続する辺はスキップ
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                new_distance = calculate_total_distance(new_path, distance_matrix)
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_path = new_path
                    improved = True
        path = best_path
    return best_distance, best_path

# 可視化
def plot_tsp(coords: List[Tuple[int, int]], path: List[int], title: str) -> None:
    plt.figure(figsize=(10, 5))
    for i in range(len(path)):
        start = coords[path[i]]
        end = coords[path[(i + 1) % len(path)]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'bo-')
        plt.annotate(f'{path[i]}', (start[0], start[1]), textcoords="offset points", xytext=(0, 10), ha='center')

        # 矢印を中間に表示
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        dx = (end[0] - start[0]) / 2
        dy = (end[1] - start[1]) / 2
        plt.annotate("", xy=(mid_x + dx, mid_y + dy), xytext=(mid_x - dx, mid_y - dy),
                    arrowprops=dict(arrowstyle="->", color="green"))

    plt.scatter(*zip(*(coords)), c='red')
    plt.title(title)
    plt.xlabel('X座標')
    plt.ylabel('Y座標')
    plt.grid(True)

# 20個の都市の座標をランダムに生成
coords = [(random.randint(0, 10), random.randint(0, 10)) for _ in range(20)]

# 距離行列の生成
distance_matrix = create_distance_matrix(coords)

# 初期巡回路（ランダム）
initial_path = list(range(len(coords)))
random.shuffle(initial_path)

# 初期解の計算
start_time = time.time()
initial_distance = calculate_total_distance(initial_path, distance_matrix)
initial_time = time.time() - start_time
print(f"初期解の総距離: {initial_distance}")
print(f"経路: {initial_path}")
print(f"初期解の計算時間: {initial_time:.6f} 秒")
plot_tsp(coords, initial_path, "初期解")

# 2-optで巡回路を改善
start_time = time.time()
opt_distance, opt_path = two_opt(initial_path, distance_matrix)
opt_time = time.time() - start_time
print(f"2-opt法の最短距離: {opt_distance}")
print(f"経路: {opt_path}")
print(f"2-opt法の計算時間: {opt_time:.6f} 秒")
plot_tsp(coords, opt_path, "2-opt法によるTSP解")

plt.show()