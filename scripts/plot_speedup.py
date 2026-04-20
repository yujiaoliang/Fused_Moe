"""Generate speedup bar chart for paper.tex Table 3 data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from Table 3 (Round 15 peak session)
tokens = [1, 7, 15, 16, 32, 52, 57, 62, 80, 901, 8192, 14107]
speedup = [None, 95.85, 106.65, 72.44, 51.59, 61.82, 50.66, 55.52, 48.09, 23.04, 10.10, 9.00]
labels = ['1', '7', '15', '16', '32', '52', '57', '62', '80', '901', '8K', '14K']

# Color by bottleneck regime
colors = []
for t, s in zip(tokens, speedup):
    if s is None:
        colors.append('#9e9e9e')       # T=1 (no speedup reported)
    elif t <= 80:
        colors.append('#1976d2')       # GEMM1-bound (blue)
    else:
        colors.append('#d32f2f')       # GEMM2-bound (red)

# Replace None with 0 for plotting
speedup_plot = [s if s is not None else 0 for s in speedup]

fig, ax = plt.subplots(figsize=(7, 3.2))

bars = ax.bar(range(len(labels)), speedup_plot, color=colors, edgecolor='white', linewidth=0.5, width=0.75)

# Add value labels on bars
for i, (bar, s) in enumerate(zip(bars, speedup)):
    if s is not None:
        va = 'bottom'
        y = bar.get_height() + 1.5
        fontsize = 7.5
        if s < 15:
            fontsize = 7
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f'{s:.1f}x', ha='center', va=va, fontsize=fontsize, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, 2,
                'N/A', ha='center', va='bottom', fontsize=7, color='#666')

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_xlabel('Token count ($T$)', fontsize=10)
ax.set_ylabel('Speedup over baseline', fontsize=10)

# Mean line
valid = [s for s in speedup if s is not None]
mean_val = np.mean(valid)
ax.axhline(y=mean_val, color='#388e3c', linestyle='--', linewidth=1.2, alpha=0.8)
ax.text(len(labels) - 0.5, mean_val + 2, f'Mean: {mean_val:.1f}x',
        fontsize=8, color='#388e3c', ha='right', fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1976d2', label='GEMM1-bound (small/medium $T$)'),
    Patch(facecolor='#d32f2f', label='GEMM2-bound (large $T$)'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='upper right', framealpha=0.9)

ax.set_ylim(0, 120)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=9)

plt.tight_layout()
plt.savefig('D:/Janice_Research/2026/mlsys_note/figures/speedup_bar.pdf', bbox_inches='tight', dpi=300)
plt.savefig('D:/Janice_Research/2026/mlsys_note/figures/speedup_bar.png', bbox_inches='tight', dpi=300)
print("Saved figures/speedup_bar.pdf and .png")
