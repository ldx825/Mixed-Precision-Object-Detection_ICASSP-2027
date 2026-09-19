"""24_make_fig_reversal.py — camera-ready regeneration of Figure 1 (vector).

Two panels:
  (a) singleton-vs-conditional pairwise order reversal rate vs context size,
      raw vs large-gap (|gap| > 0.002) filtered;
  (b) top-1 mismatch rate vs context size.

Style: colorblind-safe palette, large fonts, vector PDF for paper inclusion.
Data: outputs/Exp31_context_rank_reversal/summary.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

d = json.load(open(ROOT / "outputs/Exp31_context_rank_reversal/summary.json"))
S = [1, 2, 3]
raw = [d[f"L{i}"]["reversal_rate_pooled"] * 100 for i in S]
filt = [d[f"L{i}"]["reversal_rate_thr_pooled"] * 100 for i in S]
top1 = [d[f"L{i}"]["top1_mismatch_frac"] * 100 for i in S]

plt.rcParams.update({
    "pdf.fonttype": 42,  # embed TrueType (Type 42), never Type 3 (ICASSP requirement)
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
})
BLUE, ORANGE, GREEN = "#0173B2", "#DE8F05", "#029E73"

fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.5))
x = np.arange(3)

# ---- (a) reversal rate ----
ax = axes[0]
w = 0.36
b1 = ax.bar(x - w / 2, raw, w, color=BLUE, edgecolor="white",
            linewidth=0.35, label="all pairs")
b2 = ax.bar(x + w / 2, filt, w, color=ORANGE, edgecolor="white",
            linewidth=0.35, label="$|$gap$|>0.002$")
for bars, vals in ((b1, raw), (b2, filt)):
    for r, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", (r.get_x() + r.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 1.4),
                    ha="center", fontsize=6.0, color="0.25")
ax.set_xticks(x)
ax.set_xticklabels(["1", "2", "3"])
ax.set_xlabel("context size $|S|$")
ax.set_ylabel("reversal rate (\\%)")
ax.set_ylim(0, 37)
ax.legend(frameon=False, loc="upper left", ncols=2, handlelength=0.9,
          handletextpad=0.4, columnspacing=0.8, borderpad=0.1,
          labelspacing=0.25)

# ---- (b) top-1 mismatch ----
ax = axes[1]
b = ax.bar(x, top1, 0.5, color=GREEN, edgecolor="white", linewidth=0.35)
for r, v in zip(b, top1):
    ax.annotate(f"{v:.0f}", (r.get_x() + r.get_width() / 2, v),
                textcoords="offset points", xytext=(0, 1.4),
                ha="center", fontsize=6.0, color="0.25")
ax.set_xticks(x)
ax.set_xticklabels(["1", "2", "3"])
ax.set_xlabel("context size $|S|$")
ax.set_ylabel("top-1 mismatch (\\%)")
ax.set_ylim(0, 78)

fig.tight_layout(pad=0.25)
fig.savefig(FIGDIR / "fig1_reversal.pdf")
fig.savefig(FIGDIR / "fig1_reversal.png", dpi=300)
print("wrote", FIGDIR / "fig1_reversal.pdf")
print("raw:", [round(v, 2) for v in raw], "filt:", [round(v, 2) for v in filt],
      "top1:", [round(v, 1) for v in top1])
