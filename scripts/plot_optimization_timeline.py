"""Generate a representative cumulative optimization timeline aligned with the README phases."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parents[1] / "figures"

phases = [
    "PyTorch eager\nbaseline",
    "Monolithic\nTriton",
    "Core Triton\noptimizations",
    "Bucket\nspecialization",
    "Per-T isolation\nand final tuning",
]
peaks = [1.0, 12.9, 46.0, 55.0, 56.0]

fig, ax = plt.subplots(figsize=(7, 3.2))
x = np.arange(len(phases))
colors_fill = ["#9e9e9e"] + ["#1976d2"] * (len(phases) - 1)

for i in range(len(phases)):
    if i == 0:
        bottom = 0.0
        height = peaks[0]
    else:
        bottom = peaks[i - 1]
        height = peaks[i] - peaks[i - 1]

    ax.bar(x[i], height, bottom=bottom, width=0.62, color=colors_fill[i], edgecolor="white", linewidth=0.5, alpha=0.88)
    ax.text(x[i], peaks[i] + 1.0, f"{peaks[i]:.1f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")
    if height > 4:
        ax.text(x[i], bottom + height / 2, f"+{height:.1f}x", ha="center", va="center", fontsize=7, color="white", fontweight="bold")

for i in range(len(phases) - 1):
    ax.plot([x[i] + 0.31, x[i + 1] - 0.31], [peaks[i], peaks[i]], color="#666", linewidth=0.8, linestyle="--", alpha=0.45)

ax.set_xticks(x)
ax.set_xticklabels(phases, fontsize=7.5)
ax.set_ylabel("Representative peak speedup", fontsize=10)
ax.set_ylim(0, 62)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "optimization_timeline.pdf", bbox_inches="tight", dpi=300)
plt.savefig(OUT_DIR / "optimization_timeline.png", bbox_inches="tight", dpi=300)
print("Saved figures/optimization_timeline.pdf and .png")
