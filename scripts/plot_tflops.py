"""Generate a representative TFLOPS trend chart aligned with the README summary."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parents[1] / "figures"

# Representative throughput trend from README + reference profiler notes.
T_vals = [1, 7, 14, 16, 32, 52, 80, 901, 11948, 14107]
gemm1_tf = [92, 140, 166, 170, 172, 164, 154, 495, 520, 535]
gemm2_tf = [29, 105, 116, 124, 140, 148, 150, 180, 216, 216]

fp8_peak = 2500
tf32_peak = 1250

fig, ax = plt.subplots(figsize=(7, 3.2))

ax.plot(range(len(T_vals)), gemm1_tf, "o-", color="#1976d2", linewidth=1.8, markersize=5, label="GEMM1 (FP8 dot)", zorder=3)
ax.plot(range(len(T_vals)), gemm2_tf, "s-", color="#d32f2f", linewidth=1.8, markersize=5, label="GEMM2 (FP16 A-side)", zorder=3)

ax.axhline(y=fp8_peak, color="#1976d2", linestyle=":", linewidth=1, alpha=0.35)
ax.text(len(T_vals) - 0.4, fp8_peak + 30, f"FP8 peak: {fp8_peak}T", fontsize=7, color="#1976d2", ha="right", alpha=0.6)
ax.axhline(y=tf32_peak, color="#d32f2f", linestyle=":", linewidth=1, alpha=0.35)
ax.text(len(T_vals) - 0.4, tf32_peak + 30, f"TF32 peak: {tf32_peak}T", fontsize=7, color="#d32f2f", ha="right", alpha=0.6)

ax.axvspan(-0.5, 6.5, alpha=0.06, color="#1565c0", zorder=0)
ax.axvspan(6.5, 7.5, alpha=0.05, color="#2e7d32", zorder=0)
ax.axvspan(7.5, 9.5, alpha=0.06, color="#b71c1c", zorder=0)
ax.text(3.2, 120, "Memory-bound\n(small / mid $T$)", fontsize=8, ha="center", color="#1565c0", style="italic")
ax.text(7.0, 120, "Transition", fontsize=8, ha="center", color="#2e7d32", style="italic")
ax.text(8.5, 120, "Hybrid large-$T$", fontsize=8, ha="center", color="#b71c1c", style="italic")

ax.set_xticks(range(len(T_vals)))
ax.set_xticklabels([str(t) if t < 1000 else str(t) for t in T_vals], fontsize=8, rotation=45, ha="right")
ax.set_xlabel("Token count ($T$)", fontsize=10)
ax.set_ylabel("Representative effective TFLOPS", fontsize=10)
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.74, 0.60), framealpha=0.92)
ax.set_ylim(0, 2700)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "tflops_efficiency.pdf", bbox_inches="tight", dpi=300)
plt.savefig(OUT_DIR / "tflops_efficiency.png", bbox_inches="tight", dpi=300)
print("Saved figures/tflops_efficiency.pdf and .png")
