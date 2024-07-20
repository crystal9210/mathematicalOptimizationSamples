import itertools # イテレータを生成するための関数を提供するパッケージ、組み合わせや順列の計算に利用
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple # 型ヒント(Type Hints)を提供するためのパッケージ、List[int]:整数型のリスト、Tuple[int, int]:二つの整数を含むタプルを意味、Pythonの型アノテーションを提供することで関数や変数の型安全性を実現
import random
import time

# 日本語フォントの設定（Macの標準日本語フォントを使用）
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['font.sans-serif'] = ['Hiragino Sans']

# 距離行列の作成
# ndarray:numpyの配列、coords: List[Tuple[int, int]]=coordsの型アノテーションを付与して型安全性を提供、今回はcoordsが(x,y)の整数の位置を保持する整数のペアのリストであることを意味 | def 関数名(引数)->返り値の型:(以降の行でインテンドを入れて内部処理)
def create_distance_matrix(coords: List[Tuple[int, int]]) -> np.ndarray:
    n = len(coords) # 都市の数をnに格納
    distance_matrix = np.zeros((n, n)) # 初期値0でn*nの行列を初期化、インスタンス生成
    for i in range(n):
        for j in range(n):
            # normの引数内部の処理ー＞例：(1,0)-(3,5)=(-2,-5) ー＞ linalg(線形代数)モジュールのnorm関数により引数として与えられた配列のユークリッドノルム(ベクトルの大きさ)を計算
            distance_matrix[i, j] = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
    return distance_matrix

# 巡回路の距離を計算
def calculate_total_distance(path: List[int], distance_matrix: np.ndarray) -> float:
    # path:都市の順回路を示すリスト、例:path=[0, 3, 2, 1] なら 0->3->2->1->0という順序で巡回する | % len(path)をインデックス計算に含めることで巡回することを保証
    return sum(distance_matrix[path[i], path[(i + 1) % len(path)]] for i in range(len(path)))

# 2-optによる巡回路の改善
# ポイント：forループでインデックスを一方向で回しているため、二つのインデックスを固定するごとに反転した処理が巡回路の求めるパフォーマンスを改善するのであれば更新し、そうでなければ更新せずに
# ループを続けるが、これは大幅な数値改善は期待できるが、総当たり的な戦略と同等の最適解を求められるというわけではないことに注意
# 引数: path:都市の順回路を示すリスト、distance_matrix:都市間の距離行列、戻り値:最適化された順回路の総距離と巡回路(タプル)
def two_opt(path: List[int], distance_matrix: np.ndarray) -> Tuple[float, List[int]]:
    best_distance = calculate_total_distance(path, distance_matrix)
    best_path = path.copy() # 現在の巡回路をコピーして格納
    improved = True # 巡回路が改善されたかどうかを示すフラグの初期化
    # forループで網羅的に更新戦略を適用しているが、一通り走査および適用が終了した後、再度同様の処理で改善がされるかもしれないのでwhileでimprovedフラグを用いてそれが走査後Falseとなるまで二重forループによる更新処理を複数回繰り返す
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path)):
                if j - i == 1: continue  # 連続する辺はスキップ
                # 新しい巡回路の生成ー＞2-opt戦略のロジックに一致 | path[:i]はi以前の部分、path[i:j][::-1]はiからjまでの部分を逆順にしたもの、path[j:]はj以降のインデックスの部分
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                # 新しい巡回路の距離の計算 | ポイント：distance_matrixは都市間の距離を固定的に格納しており、新しい順路に対する巡回路の総距離を計算するためのツールとして利用
                new_distance = calculate_total_distance(new_path, distance_matrix)
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_path = new_path
                    improved = True
        path = best_path # ここでのbest_pathは上述の処理におけるループにおけるbest_pathの結果を格納したもの、ここで、更新することで後続の処理において今回の結果を反映することができる
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