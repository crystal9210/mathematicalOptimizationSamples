# 局所探索法を用いて与えられた関数の最小値を探索するプログラム
# 局所探索法の基本ステップ
# 局所探索法は以下のステップで進行する。
# 初期解を決定: 適当な初期解を設定
# 近傍の探索: 現在の解の近くにある解（近傍）を生成
# 改善解の確認: 近傍の中から、現在の解よりも良い解（目的関数の値が小さい解）があるか確認
# 解の更新: 改善解が見つかればそれを新しい解として設定し、2のステップに戻る、改善解が見つからなければ終了する。


# 局所探索法とは：適当な初期解から出発し、解の近傍にそれより良い解があれば置き換える、という操作を繰り返し実行して、解の更新が行われなくなった時終了する
# ー＞TODO:ここで、「解の更新が行われなくなった」の判断基準としてはどのような指標を置くのか

# 近傍回の妥当性ー＞「良い解同士は似た構造を持つ」；近傍回最適性(proximity optimality principle,POP)
# 挿入近傍、交換近傍
import numpy as np
import matplotlib.pyplot as plt
from typing import List # 型ヒントのためのライブラリ、リストの中に何が含まれているかを明示するために使用

# 引数にfloat値を受け取りそれに対する一定の他公式の数値計算の結果を計算・返す関数
def f(x: float) -> float:
    x = np.clip(x, -3.3, 3.3) # np.clip:xを第2引数と第3引数の間の範囲に収める
    # xを使って多項式を計算し、結果として返す
    return (44 * x**6 +13 * x**5 - 638 * x**4 -88 * x**3 +2600 * x**2 -261 * x + 6.5653624787847)

# 可視化関数
# hists:探索履歴のリスト、draw_arrow:履歴間の遷移を矢印で書くかどうかのフラグ
def vis(hists: List[List[float]], draw_arrow: bool = False) -> None:
    xs = np.linspace(-3.3, 3.3, 100) # linspace:-3.3から3.3まで等間隔で100個の点を生成する関数
    ys=f(xs) # 生成した点に対して目的関数fを適用して、その数値を計算、取得
    fig = plt.figure(figsize=(12, 5)) # 図の作成、figsize:図のサイズを指定
    # サブプロット関数の挙動の確認ファイルー＞同一ディレクトリのsabplot.pyファイルを実行して確認
    ax = fig.add_subplot(111) # サブプロットの作成；111は1行1列の1番目のプロットを意味
    ax.plot(xs, ys)
    ax.set_title("f(x) (-3.3 <= x <= 3.3)")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xticks(np.linspace(-3.3, 3.3, 5)) # -3.3から3.3の範囲で等間隔に5つの点を生成するー＞今回：[-3.3, -1.65, 0, 1.65, 3.3]ー＞X軸のメモリとして表示
    ax.grid() # グリッド線の表示
    # 探索履歴ごとにループ
    # 例：histsが[[x1, x2, ...], [y1, y2, ...]]ー＞最初のループはhistが[x1, x2, ...]、次のループは[y1, y2, ...]となる
    for hist in hists:
        # TODO:下のrangeの第三引数の値を変えて観察する、1-5の整数とかが良いかも。
        for i in range(1, len(hist)-1, 5): # 1からlen(hist)-1までの範囲でステップ幅を5としてインデックスを生成；例：hist:20ー＞1,6,11,16のインデックスを生成、ループー＞全てのポイントをプロットせず、5ステップごとに間引くことでプロットが過密にならず見やすくなる
            ax.plot(hist[i], f(hist[i]), 'o', color= "gray") # 灰色のマーカで間引いた各点をプロット
        ax.plot(hist[0], f(hist[0]), 'o', color="blue")
        ax.plot(hist[-1], f(hist[-1]), 'o', color="red") # hist[-1]:探索の最終点
    if draw_arrow:
        for hist in hists:
            for j in range(len(hist) - 1):
                ax.annotate("",
                            xy=(hist[j + 1], f(hist[j + 1])),
                            xytext=(hist[j], f(hist[j])),
                            arrowprops=dict(arrowstyle="->", color="green"))

    plt.show()

# 局所探索方の関数実装 | x0:初期解
def local_search(x0: float) -> List[float]:
    x0 = np.clip(x0, -3.3, 3.3) # とりあえず初期解の範囲を限定ー＞TODO:この範囲指定は今回どういう感じの意図があってしているのか
    x = x0
    k = 0 # 探索回数初期化
    best_score = f(x) # 初期解の目的間数値を計算して代入
    hist = [x] # 初期解を探索履歴に追加

    while k < 100:
        k += 1
        Nx = np.linspace(x-0.05, x+0.05, 10) # 近傍を生成
        fNx = f(Nx) # 近傍の各点の目的間数値を計算
        if np.any(fNx < best_score): # np.any:近傍に現在の解よりも良い解があるか確認ー＞与えられた配列内で少なくとも一つの要素が指定された条件を満たすかどうかを判定
            new_x = np.random.choice(Nx[np.where(fNx < best_score)[0]])
            best_score = f(new_x)
            x = new_x
            hist.append(new_x)
        else:
            # 改善解が無い場合、探索を終了
            break

    assert(len(hist) == k)
    print(f"探索回数: {len(hist)}回")
    print(f"得られた最適値: {f(hist[-1])}")
    print("真の最適値: 0")
    return hist


# 実行ルーチン
hist =local_search(x0=-2)
vis([hist],draw_arrow=True)