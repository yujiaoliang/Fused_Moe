"""Stacked bar chart: per-kernel time breakdown across T values."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from ncu_profiler_yjl.txt SUMMARY table (ms)
T_vals =   [1,     7,     14,    15,    16,    32,    52,    62,    80,    901,   11948,  14107]
gemm1  =   [0.031, 0.047, 0.067, 0.046, 0.071, 0.116, 0.094, 0.096, 0.168, 0.269, 1.343,  1.988]
gemm2  =   [0.049, 0.031, 0.049, 0.030, 0.049, 0.058, 0.048, 0.048, 0.072, 0.201, 0.748,  1.110]
routing=   [0.012, 0.006, 0.007, 0.008, 0.008, 0.007, 0.008, 0.009, 0.008, 0.016, 0.000,  0.000]
sorting=   [0.000, 0.002, 0.002, 0.003, 0.003, 0.003, 0.003, 0.003, 0.005, 0.024, 0.002,  0.002]
# "other" = total CUDA - gemm1 - gemm2 - routing - sorting
cuda_total=[0.095, 0.093, 0.132, 0.092, 0.136, 0.191, 0.160, 0.164, 0.261, 0.530, 2.432,  3.503]
other  =   [ct - g1 - g2 - r - s for ct, g1, g2, r, s in zip(cuda_total, gemm1, gemm2, routing, sorting)]

labels = ['1', '7', '14', '15', '16', '32', '52', '62', '80', '901', '12K', '14K']
x = np.arange(len(labels))
width = 0.65

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0), gridspec_kw={'width_ratios': [9, 3], 'wspace': 0.35})

# Left: absolute time (stacked)
colors = ['#1976d2', '#d32f2f', '#4caf50', '#ff9800', '#9e9e9e']
names = ['GEMM1', 'GEMM2', 'Routing', 'Sorting', 'Other']
bottom = np.zeros(len(labels))
for data, color, name in zip([gemm1, gemm2, routing, sorting, other], colors, names):
    ax1.bar(x, data, width, bottom=bottom, color=color, label=name, edgecolor='white', linewidth=0.3)
    bottom += np.array(data)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
ax1.set_ylabel('CUDA kernel time (ms)', fontsize=9)
ax1.set_xlabel('Token count ($T$)', fontsize=9)
ax1.legend(fontsize=7, ncol=3, loc='upper left', framealpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='y', labelsize=8)
ax1.set_title('(a) Absolute kernel time', fontsize=9, fontweight='bold')

# Right: percentage (stacked, for representative T values)
sel_idx = [1, 4, 8, 9, 11]  # T=7, 16, 80, 901, 14107
sel_labels = [labels[i] for i in sel_idx]
sel_x = np.arange(len(sel_labels))

pct_data = []
for i in sel_idx:
    total = cuda_total[i]
    pct_data.append([gemm1[i]/total*100, gemm2[i]/total*100,
                     routing[i]/total*100, sorting[i]/total*100, other[i]/total*100])
pct_data = np.array(pct_data)

bottom2 = np.zeros(len(sel_labels))
for j, (color, name) in enumerate(zip(colors, names)):
    ax2.barh(sel_x, pct_data[:, j], 0.55, left=bottom2, color=color, edgecolor='white', linewidth=0.3)
    bottom2 += pct_data[:, j]

ax2.set_yticks(sel_x)
ax2.set_yticklabels([f'$T$={l}' for l in sel_labels], fontsize=8)
ax2.set_xlabel('Kernel time (%)', fontsize=9)
ax2.set_xlim(0, 100)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='x', labelsize=8)
ax2.set_title('(b) Percentage breakdown', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('D:/Janice_Research/2026/mlsys_note/figures/time_breakdown.pdf', bbox_inches='tight', dpi=300)
plt.savefig('D:/Janice_Research/2026/mlsys_note/figures/time_breakdown.png', bbox_inches='tight', dpi=300)
print("Saved figures/time_breakdown.pdf and .png")
