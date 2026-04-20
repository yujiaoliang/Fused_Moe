"""Waterfall chart: cumulative optimization impact."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

phases = [
    'PyTorch eager\nbaseline',
    'Monolithic\nTriton',
    'FP8 native\ndot (GEMM1)',
    'Non-atomic\nGEMM2',
    'Bucket\nspecialization',
    'FP16\nintermediate',
    'T=901 +\nmedium-T tune',
]
peaks = [1.0, 12.9, 26.0, 47.9, 88.0, 96.0, 106.65]

fig, ax = plt.subplots(figsize=(7, 3.2))

# Draw waterfall bars
colors_fill = ['#9e9e9e'] + ['#1976d2'] * 6
x = np.arange(len(phases))

# Each bar: bottom = previous peak, height = delta
for i in range(len(phases)):
    if i == 0:
        bottom = 0
        height = peaks[0]
    else:
        bottom = peaks[i-1]
        height = peaks[i] - peaks[i-1]

    bar = ax.bar(x[i], height, bottom=bottom, width=0.6,
                 color=colors_fill[i], edgecolor='white', linewidth=0.5, alpha=0.85)

    # Label at top of cumulative bar
    ax.text(x[i], peaks[i] + 1.5, f'{peaks[i]:.1f}x',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Delta label inside bar (if tall enough)
    if height > 8:
        ax.text(x[i], bottom + height/2, f'+{height:.1f}x',
                ha='center', va='center', fontsize=7, color='white', fontweight='bold')

# Connect bars with lines
for i in range(len(phases) - 1):
    ax.plot([x[i] + 0.3, x[i+1] - 0.3], [peaks[i], peaks[i]],
            color='#666', linewidth=0.8, linestyle='--', alpha=0.5)

ax.set_xticks(x)
ax.set_xticklabels(phases, fontsize=7.5)
ax.set_ylabel('Peak speedup', fontsize=10)
ax.set_ylim(0, 120)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=9)

plt.tight_layout()
plt.savefig('D:/Janice_Research/2026/mlsys_note/figures/optimization_timeline.pdf', bbox_inches='tight', dpi=300)
plt.savefig('D:/Janice_Research/2026/mlsys_note/figures/optimization_timeline.png', bbox_inches='tight', dpi=300)
print("Saved figures/optimization_timeline.pdf and .png")
