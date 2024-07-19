import itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
import matplotlib.font_manager as fm
import random

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

# 総当たり法によるTSP解法
def solve_tsp_brute_force(distance_matrix: np.ndarray) -> Tuple[float, List[int]]:
    n = len(distance_matrix)
    min_path = None
    min_distance = float('inf')
    for perm in itertools.permutations(range(n)):
        current_distance = sum(distance_matrix[perm[i], perm[i+1]] for i in range(n-1))
        current_distance += distance_matrix[perm[-1], perm[0]]
        if current_distance < min_distance:
            min_distance = current_distance
            min_path = perm
    return min_distance, list(min_path)

# グリーディ法によるTSP解法
def solve_tsp_greedy(distance_matrix: np.ndarray) -> Tuple[float, List[int]]:
    n = len(distance_matrix)
    current_city = 0
    unvisited = set(range(1, n))
    path = [current_city]
    total_distance = 0
    while unvisited:
        next_city = min(unvisited, key=lambda city: distance_matrix[current_city, city])
        total_distance += distance_matrix[current_city, next_city]
        current_city = next_city
        path.append(current_city)
        unvisited.remove(current_city)
    total_distance += distance_matrix[path[-1], path[0]]
    return total_distance, path

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

    plt.scatter(*zip(*coords), c='red')
    plt.title(title)
    plt.xlabel('X座標')
    plt.ylabel('Y座標')
    plt.grid(True)

# 都市の座標
# coords = [(0, 0), (1, 3), (4, 3), (6, 1), (3, 0), (3, 5)]
# 100個の都市の座標をランダムに生成ー＞総当たり法での計算時間が現実的じゃないので事実上無理、ここで、10個に減らしてできるか試す(29個でも数時間から数日かかるらしい...)
# 下のデータセットを使うと計算量の違いがよくわかる、そして、計算量が大きくなったときトレードオフ的にどの程度許容されるのかは今後学んでいくうちに...
coords = [(random.randint(0, 10), random.randint(0, 10)) for _ in range(10)]

# 距離行列の生成
distance_matrix = create_distance_matrix(coords)

# 総当たり法で解く | 計算量:O(n!)
start_time = time.time()
bf_distance, bf_path = solve_tsp_brute_force(distance_matrix)
bf_time = time.time() - start_time
print(f"総当たり法の最短距離: {bf_distance}")
print(f"経路: {bf_path}")
print(f"総当たり法の計算時間: {bf_time:.6f} 秒")
plot_tsp(coords, bf_path, "総当たり法によるTSP解")

# グリーディ法で解く |
# 計算量:O(n^2)->各ステップで局所的に最適な選択をするため、各都市に対して最も近い都市を見つける
start_time = time.time()
greedy_distance, greedy_path = solve_tsp_greedy(distance_matrix)
greedy_time = time.time() - start_time
print(f"グリーディ法の最短距離: {greedy_distance}")
print(f"経路: {greedy_path}")
print(f"グリーディ法の計算時間: {greedy_time:.6f} 秒")
plot_tsp(coords, greedy_path, "グリーディ法によるTSP解")
plt.show()