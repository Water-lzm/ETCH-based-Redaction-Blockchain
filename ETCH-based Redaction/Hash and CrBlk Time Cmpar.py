import matplotlib.pyplot as plt
import numpy as np

# 数据
t_values = [3, 9, 15, 21, 27, 33]
sha256_total = [1.834, 1.877, 2.101, 2.263, 2.357, 2.261]
etch_total = [1.979, 1.998, 2.183, 2.448, 2.464, 2.349]

sha256_hash = [0.358, 0.356, 0.359, 0.379, 0.396, 0.361]
etch_hash = [0.612, 0.61, 0.629, 0.672, 0.662, 0.646]

double_sha256_total = [2 * x for x in sha256_total]
double_sha256_hash = [2 * x for x in sha256_hash]

# 样式设定：更符合论文出版风格
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2,
    'lines.markersize': 6
})

# 颜色设置
color_sha = "orange"    # SHA-256颜色
color_etch = "purple"   # ETCH颜色
color_ref = "gray"      # 参考线颜色

# 图1：单次哈希开销
plt.figure(figsize=(6, 4.5))
plt.plot(t_values, sha256_hash, marker='o', color=color_sha, label='SHA-256', markersize=4)
plt.plot(t_values, etch_hash, marker='s', color=color_etch, label='ETCH', markersize=4)
plt.plot(t_values, double_sha256_hash, linestyle='--', color=color_ref, label='2× SHA-256', markersize=4)

#plt.title("Per-Hash Computation Overhead")
plt.xlabel("Number of Redactors")
plt.ylabel("Time (ms)")
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# 图2：总区块构建开销
plt.figure(figsize=(6, 4.5))
plt.plot(t_values, sha256_total, marker='o', color=color_sha, label='SHA-256', markersize=4)
plt.plot(t_values, etch_total, marker='s', color=color_etch, label='ETCH', markersize=4)
plt.plot(t_values, double_sha256_total, linestyle='--', color=color_ref, label='2× SHA-256', markersize=4)

#plt.title("Block Creation Overhead (100-block Average)")
plt.xlabel("Number of Redactors")
plt.ylabel("Time (ms)")
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
