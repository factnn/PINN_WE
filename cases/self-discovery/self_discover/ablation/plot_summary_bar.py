#!/usr/bin/env python3
"""Ablation summary bar chart: max alpha error for each method."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "output"

methods = [
    ("123 (Ours)",              3.7e-6,   "#1f77b4"),
    ("1234 (+warmup)",          3.8e-6,   "#5dade2"),
    ("12(3-sonic)4",            4.5e-5,   "#27ae60"),
    ("12(3-shock)4",            4.2e-3,   "#2ecc71"),
    ("12(3-soft)4",             3.8e-3,   "#f39c12"),
    ("1/34 (divided+warmup)",   1.4e-5,   "#48c9b0"),
    ("xi-23 (xi, no warmup)",   2.0e-3,   "#af7ac5"),
    ("xi-234 (xi+warmup)",      1.36e-2,  "#8e44ad"),
    ("12 (base, no constraint)",5.8e-1,   "#95a5a6"),
    ("12+4 (no constraint)",    5.2e-1,   "#7f8c8d"),
    ("1/3 (divided, no warmup)",5.2e-1,   "#e74c3c"),
    ("1x3 (mult, no warmup)",   9.0e-1,   "#ec7063"),
    ("xi-(3soft)4",             1.48,     "#c0392b"),
    ("1x34 (mult+warmup)",      10.13,    "#922b21"),
]

fig, ax = plt.subplots(figsize=(11, 7))

labels = [m[0] for m in methods]
errors = [m[1] for m in methods]
colors = [m[2] for m in methods]

y_pos = np.arange(len(methods))

bars = ax.barh(y_pos, errors, color=colors, edgecolor="black", linewidth=0.5, height=0.7)

bars[0].set_linewidth(2.5)
bars[0].set_edgecolor("black")
bars[1].set_linewidth(2.5)
bars[1].set_edgecolor("black")

ax.set_xscale("log")
ax.set_xlim(1e-7, 50)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Max alpha relative error across 4 configs (%)", fontsize=12)
ax.set_title("Ablation Study: Impact of Each Design Component", fontsize=14, fontweight="bold")

ax.axvline(1e-3, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(1e-3, len(methods)-0.3, "0.001%", fontsize=8, color="gray", ha="center")
ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(1.0, len(methods)-0.3, "1%", fontsize=8, color="gray", ha="center")

for i, (lbl, err, _) in enumerate(methods):
    if err < 0.01:
        ax.text(err * 2.0, i, f"{err:.1e}%", va="center", fontsize=8, color="black")
    else:
        ax.text(err * 1.5, i, f"{err:.2f}%", va="center", fontsize=8, color="black")

ax.axvspan(1e-7, 1e-4, alpha=0.05, color="green")
ax.axvspan(1e-1, 50, alpha=0.05, color="red")

ax.grid(True, alpha=0.2, axis="x")
fig.tight_layout()
fig.savefig(OUT / "fig_ablation_bar.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT / 'fig_ablation_bar.png'}")
