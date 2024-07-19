import numpy as np
import matplotlib.pyplot as plt

# データ生成
x = np.linspace(0, 10, 100)

# サブプロットの生成
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# サブプロット 1: y = x のプロット
axs[0, 0].plot(x, x)
axs[0, 0].set_title('y = x')
axs[0, 0].grid()

# サブプロット 2: y = x^2 のプロット
axs[0, 1].plot(x, x**2)
axs[0, 1].set_title('y = x^2')
axs[0, 1].grid()

# サブプロット 3: y = sin(x) のプロット
axs[1, 0].plot(x, np.sin(x))
axs[1, 0].set_title('y = sin(x)')
axs[1, 0].grid()

# サブプロット 4: y = cos(x) のプロット
axs[1, 1].plot(x, np.cos(x))
axs[1, 1].set_title('y = cos(x)')
axs[1, 1].grid()

# サブプロット間のレイアウト調整
plt.tight_layout()

# グラフの表示
plt.show()
