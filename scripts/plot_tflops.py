"""TFLOPS efficiency vs T for GEMM1 and GEMM2."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# From ncu_profiler_yjl.txt summary
T_vals = [1, 7, 14, 15, 16, 32, 52, 62, 80, 901, 11948, 14107]
gemm1_tf = [91.0, 139.4, 167.7, 102.1, 173.1, 178.1, 169.4, 176.3, 156.6, 502.4, 537.4, 540.7]
gemm2_tf = [29.0, 105.2, 115.4, 78.0, 125.3, 178.7, 165.7, 174.4, 182.4, 337.3, 482.1, 484.3]

# B200 peaks
fp8_peak = 2500
tf32_peak = 1250

fig, ax = plt.subplots(figsize=(7, 3.2))

ax.plot(range(len(T_vals)), gemm1_tf, 'o-', color='#1976d2', linewidth=1.8,
        markersize=5, label='GEMM1 (FP8 dot)', zorder=3)
ax.plot(range(len(T_vals)), gemm2_tf, 's-', color='#d32f2f', linewidth=1.8,
        markersize=5, label='GEMM2 (FP16 A-side)', zorder=3)

# Peak reference lines
ax.axhline(y=fp8_peak, color='#1976d2', linestyle=':', linewidth=1, alpha=0.4)
ax.text(len(T_vals)-0.5, fp8_peak+30, f'FP8 peak: {fp8_peak}T', fontsize=7,
        color='#1976d2', ha='right', alpha=0.6)
ax.axhline(y=tf32_peak, color='#d32f2f', linestyle=':', linewidth=1, alpha=0.4)
ax.text(len(T_vals)-0.5, tf32_peak+30, f'TF32 peak: {tf32_peak}T', fontsize=7,
        color='#d32f2f', ha='right', alpha=0.6)

# Regions
ax.axvspan(-0.5, 8.5, alpha=0.06, color='blue', zorder=0)
ax.axvspan(8.5, 11.5, alpha=0.06, color='red', zorder=0)
ax.text(4, 50, 'Memory-bound\n(small/medium $T$)', fontsize=8, ha='center',
        color='#1565c0', style='italic')
ax.text(10, 50, 'Compute-bound\n(large $T$)', fontsize=8, ha='center',
        color='#b71c1c', style='italic')

ax.set_xticks(range(len(T_vals)))
ax.set_xticklabels([str(t) if t < 1000 else f'{t//1000}K' for t in T_vals],
                   fontsize=8, rotation=45, ha='right')
ax.set_xlabel('Token count ($T$)', fontsize=10)
ax.set_ylabel('Effective TFLOPS', fontsize=10)
ax.legend(fontsize=8, loc='center right')
ax.set_ylim(0, 2700)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=9)

plt.tight_layout()
plt.savefig('D:/Janice_Research/2026/mlsys_note/figures/tflops_efficiency.pdf', bbox_inches='tight', dpi=300)
plt.savefig('D:/Janice_Research/2026/mlsys_note/figures/tflops_efficiency.png', bbox_inches='tight', dpi=300)
print("Saved figures/tflops_efficiency.pdf and .png")
