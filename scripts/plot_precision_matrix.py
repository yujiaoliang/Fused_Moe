"""Precision hierarchy heatmap: pass/fail matrix across formats and buffers."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Rows: buffer types; Columns: precision formats
buffers = ['Intermediate\n(GEMM1$\\to$GEMM2)', 'expert\\_out\n(GEMM2 output)']
formats = ['FP8\n(3-bit)', 'bf16\n(7-bit)', 'FP16\n(10-bit)', 'FP32\n(23-bit)']

# Values: pass count out of 19; -1 means not tested
# [Intermediate row, expert_out row]
data = np.array([
    [0,  8,  19, 19],  # Intermediate: fp8=0/19, bf16=8/19, fp16=19/19, fp32=19/19
    [0,  19, 3,  19],  # expert_out: fp8=N/A->0, bf16=19/19, fp16=3/19(overflow), fp32=19/19
])

annotations = [
    ['0/19\nabs>10K', '8/19\nptxas fail', '19/19\n+4.5%', '19/19\nbaseline'],
    ['N/A', '19/19\n+0.4%', '3/19\noverflow', '19/19\nbaseline'],
]

# Color: green for pass, red for fail, yellow for partial
def get_color(val):
    if val >= 19:
        return '#4caf50'
    elif val >= 10:
        return '#ff9800'
    elif val > 0:
        return '#ff5722'
    else:
        return '#d32f2f'

fig, ax = plt.subplots(figsize=(5.5, 2.2))

for i in range(len(buffers)):
    for j in range(len(formats)):
        color = get_color(data[i, j])
        rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white', linewidth=2, alpha=0.75)
        ax.add_patch(rect)
        text_color = 'white' if data[i, j] < 10 else 'black'
        ax.text(j + 0.5, i + 0.5, annotations[i][j],
                ha='center', va='center', fontsize=7.5, fontweight='bold', color=text_color)

ax.set_xlim(0, len(formats))
ax.set_ylim(0, len(buffers))
ax.set_xticks([x + 0.5 for x in range(len(formats))])
ax.set_xticklabels(formats, fontsize=8.5)
ax.set_yticks([y + 0.5 for y in range(len(buffers))])
ax.set_yticklabels(buffers, fontsize=8.5)
ax.invert_yaxis()
ax.xaxis.set_ticks_position('top')
ax.tick_params(axis='both', length=0)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4caf50', alpha=0.75, label='19/19 PASS'),
    Patch(facecolor='#ff9800', alpha=0.75, label='Partial pass'),
    Patch(facecolor='#d32f2f', alpha=0.75, label='FAIL (0/19)'),
]
ax.legend(handles=legend_elements, fontsize=7, loc='lower center',
          bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)

ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('D:/Janice_Research/2026/mlsys_note/figures/precision_matrix.pdf', bbox_inches='tight', dpi=300)
plt.savefig('D:/Janice_Research/2026/mlsys_note/figures/precision_matrix.png', bbox_inches='tight', dpi=300)
print("Saved figures/precision_matrix.pdf and .png")
