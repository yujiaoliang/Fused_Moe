"""Generate a representative speedup chart aligned with the latest README summary."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


OUT_DIR = Path(__file__).resolve().parents[1] / "figures"

# Representative speedup trend from the README reference session.
# T=1 is a dedicated singleton path and is intentionally shown without a scalar speedup.
labels = ["1", "7", "14", "32", "53-59", "80", "901", "11948", "14107"]
speedup = [None, 56.0, 54.0, 50.0, 55.0, 46.0, 18.0, 13.5, 14.5]
colors = [
    "#9e9e9e",  # singleton
    "#1976d2",
    "#1976d2",
    "#1976d2",
    "#1976d2",
    "#1976d2",  # GEMM1-bound / fragmented mid-T
    "#2e7d32",  # isolated transition regime
    "#d32f2f",
    "#d32f2f",  # hybrid large-T
]

plot_vals = [v if v is not None else 0.0 for v in speedup]

fig, ax = plt.subplots(figsize=(7, 3.45))
bars = ax.bar(
    range(len(labels)),
    plot_vals,
    color=colors,
    edgecolor="white",
    linewidth=0.6,
    width=0.72,
)

for bar, val in zip(bars, speedup):
    x = bar.get_x() + bar.get_width() / 2
    if val is None:
        ax.text(x, 1.5, "specialized", ha="center", va="bottom", fontsize=7, color="#666")
    else:
        ax.text(x, bar.get_height() + 1.0, f"~{val:.1f}x", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_xlabel("Token count ($T$)", fontsize=10)
ax.set_ylabel("Representative speedup", fontsize=10)
ax.set_ylim(0, 62)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=9)

legend_elements = [
    Patch(facecolor="#9e9e9e", label="Singleton dedicated path"),
    Patch(facecolor="#1976d2", label="Small / fragmented mid-$T$"),
    Patch(facecolor="#2e7d32", label="Isolated transition regime ($T{=}901$)"),
    Patch(facecolor="#d32f2f", label="Hybrid large-$T$"),
]
ax.legend(
    handles=legend_elements,
    fontsize=7.2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=2,
    framealpha=0.92,
    columnspacing=1.2,
    handlelength=1.4,
)

plt.tight_layout(rect=(0, 0.12, 1, 1))
plt.savefig(OUT_DIR / "speedup_bar.pdf", bbox_inches="tight", dpi=300)
plt.savefig(OUT_DIR / "speedup_bar.png", bbox_inches="tight", dpi=300)
print("Saved figures/speedup_bar.pdf and .png")
