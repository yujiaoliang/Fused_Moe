"""Generate a representative kernel-time breakdown chart aligned with the README summary."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parents[1] / "figures"

# Representative percentage breakdowns from the README summary.
labels = ["$T{=}1$", "$T{=}7$--80", "$T{=}901$", "$T{=}11948$--14107"]
gemm1 = np.array([32.0, 56.0, 52.6, 40.0])
gemm2 = np.array([55.0, 34.5, 35.3, 49.5])
route_sort = np.array([10.0, 7.5, 7.9, 5.5])
reduce = np.array([0.0, 2.0, 2.3, 3.0])
other = 100.0 - gemm1 - gemm2 - route_sort - reduce

series = [
    ("GEMM1", gemm1, "#1976d2"),
    ("GEMM2", gemm2, "#d32f2f"),
    ("Routing+Sort", route_sort, "#43a047"),
    ("Reduce", reduce, "#ff9800"),
    ("Other", other, "#9e9e9e"),
]

y = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(7, 2.8))
left = np.zeros(len(labels))

for name, vals, color in series:
    ax.barh(y, vals, left=left, color=color, edgecolor="white", linewidth=0.4, height=0.58, label=name)
    left += vals

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(0, 100)
ax.set_xlabel("Kernel time (%)", fontsize=10)
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", labelsize=9)
ax.legend(fontsize=7.2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), framealpha=0.92)

plt.tight_layout()
plt.savefig(OUT_DIR / "time_breakdown.pdf", bbox_inches="tight", dpi=300)
plt.savefig(OUT_DIR / "time_breakdown.png", bbox_inches="tight", dpi=300)
print("Saved figures/time_breakdown.pdf and .png")
