import matplotlib.pyplot as plt

# 样式设定：更符合论文出版风格
plt.rcParams.update({
    'font.size': 15,
    'font.family': 'serif',
    'axes.labelsize': 18,
    'axes.titlesize': 15,
    'legend.fontsize': 14,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'lines.linewidth': 6,
    'lines.markersize': 8
})
fig, ax = plt.subplots(figsize=(6.5, 4.3))

# 更强对比度的颜色组合（色盲友好 & 高可辨识度）
colors = ['black', 'red', 'blue', 'green', 'purple', 'orange']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
labels = ['RTT=0 ms (Baseline)', 'RTT=20 ms (Local Edge)', 'RTT=50 ms (Intra-Province)',
          'RTT=100 ms (National)', 'RTT=200 ms (Asia-Pacific)', 'RTT=300 ms (Intercontinental)']

t_values = [3, 9, 15, 21, 27, 33]
data = [
    [8.1, 23.1, 37.1, 50.4, 68.3, 81.6],
    [29.325, 43.357, 57.11, 70.756, 90.213, 100.789],
    [58.837, 73.513, 87.957, 101.039, 119.719, 131.431],
    [109.203, 124.04, 136.799, 151.202, 169.181, 181.753],
    [209.088, 224.048, 237.969, 250.589, 269.02, 281.275],
    [309.329, 323.07, 337.37, 350.883, 369.445, 380.398]
]

for i in range(6):
    ax.plot(t_values, data[i], marker='o', label=labels[i], color=colors[i], linestyle=linestyles[i], linewidth=2)

ax.set_xlabel("Number of Redactors", fontsize=15)
ax.set_ylabel("Redaction Overhead (ms)", fontsize=15)
#ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
ax.legend(loc='upper left', fontsize=11, frameon=False)

plt.tight_layout()
plt.show()
