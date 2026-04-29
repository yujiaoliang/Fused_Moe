"""Generate pipeline architecture diagram."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(7, 2.0))
ax.set_xlim(-0.5, 16.5)
ax.set_ylim(-1.5, 2.2)
ax.axis('off')

# Stage boxes
stages = [
    (0.0, 'Routing', '#bbdefb', 'Triton'),
    (3.0, 'Sorting', '#bbdefb', 'Triton'),
    (6.0, 'GEMM1\n+ SwiGLU', '#bbdefb', 'FP8 dot'),
    (10.0, 'GEMM2', '#bbdefb', 'Non-atomic'),
    (14.0, 'Token\nReduce', '#bbdefb', 'Atomic-free'),
]

# Buffer boxes
buffers = [
    (8.0, 'Intermediate\n(fp16/fp32)'),
    (12.0, 'expert_out\n(bf16/fp32)'),
]

box_w, box_h = 2.4, 1.1
buf_w, buf_h = 2.0, 0.7

for i, (x, label, color, annot) in enumerate(stages):
    rect = FancyBboxPatch((x, 0), box_w, box_h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#555', linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + box_w/2, box_h/2, f'{i+1}. {label}', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#1a1a1a')
    ax.text(x + box_w/2, box_h + 0.2, annot, ha='center', va='bottom',
            fontsize=6, fontstyle='italic', color='#777')

for x, label in buffers:
    rect = FancyBboxPatch((x, -1.1), buf_w, buf_h, boxstyle="round,pad=0.05",
                          facecolor='#fff9c4', edgecolor='#999', linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x + buf_w/2, -1.1 + buf_h/2, label, ha='center', va='center',
            fontsize=6.5, color='#555')

# Arrows between stages
arrow_kw = dict(arrowstyle='->', color='#555', lw=1.5, mutation_scale=12)
# Route -> Sort
ax.annotate('', xy=(3.0, box_h/2), xytext=(box_w, box_h/2), arrowprops=arrow_kw)
# Sort -> GEMM1
ax.annotate('', xy=(6.0, box_h/2), xytext=(3.0+box_w, box_h/2), arrowprops=arrow_kw)
# GEMM1 -> Intermediate (down)
ax.annotate('', xy=(8.0+buf_w/2, -0.4), xytext=(6.0+box_w/2, 0), arrowprops=arrow_kw)
# Intermediate -> GEMM2
ax.annotate('', xy=(10.0, box_h/2), xytext=(8.0+buf_w, -1.1+buf_h/2),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.5, mutation_scale=12,
                           connectionstyle='arc3,rad=-0.3'))
# GEMM2 -> expert_out (down)
ax.annotate('', xy=(12.0+buf_w/2, -0.4), xytext=(10.0+box_w/2, 0), arrowprops=arrow_kw)
# expert_out -> Reduce
ax.annotate('', xy=(14.0, box_h/2), xytext=(12.0+buf_w, -1.1+buf_h/2),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.5, mutation_scale=12,
                           connectionstyle='arc3,rad=-0.3'))

plt.tight_layout()
plt.savefig('D:/Janice_Research/2026/mlsys_note/figures/pipeline.pdf', bbox_inches='tight', dpi=300)
plt.savefig('D:/Janice_Research/2026/mlsys_note/figures/pipeline.png', bbox_inches='tight', dpi=300)
print("Saved figures/pipeline.pdf and .png")
