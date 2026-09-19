"""Exp04 — feature distortion (D_feat) vs detection damage (ΔAP) analysis.

Reads tables/Exp03_summary.json (produced by eval_sweep.py), computes:
  - Pearson / Spearman / Kendall across (group,bit) points, for overall + AP_S/M/L
  - per-bit (6/4) and per-stage sliced correlations
  - matched-distortion pairs (|ΔD|/max < 10% but large ΔAP difference)
  - rank inversions of the sensitivity ordering
Writes outputs/Exp04/correlation_report.json + figures/exp04_scatter.png
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")


def corr_block(x, y):
    from scipy import stats

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {}
    sp = stats.spearmanr(x, y)
    kt = stats.kendalltau(x, y)
    pr = stats.pearsonr(x, y)
    return {"spearman": float(sp.statistic), "spearman_p": float(sp.pvalue),
            "kendall": float(kt.statistic), "kendall_p": float(kt.pvalue),
            "pearson": float(pr.statistic), "n": int(len(x))}


def main():
    summary = json.loads((ROOT / "tables/Exp03_summary.json").read_text())
    rows = [r for r in summary["rows"] if r.get("group") != "FP" and "dAP" in r]
    if not rows:
        print("no rows with dAP")
        return
    out_dir = ROOT / "outputs/Exp04"
    out_dir.mkdir(parents=True, exist_ok=True)

    # x candidates: overall + per-stage
    x_keys = ["feat_rel_mse_all"] + [f"feat_rel_mse_stage{i}" for i in range(4)]
    y_keys = ["dAP", "dAP50", "dAP75", "dAP_S", "dAP_M", "dAP_L"]

    report = {"n_points": len(rows), "correlations": {}}
    for xk in x_keys:
        for yk in y_keys:
            x = np.array([r.get(xk, np.nan) for r in rows], dtype=float)
            y = np.array([r.get(yk, np.nan) for r in rows], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            report["correlations"][f"{xk}__{yk}"] = corr_block(x[mask], y[mask])

    # slices: per bit
    for bits in [6, 4]:
        sub = [r for r in rows if r.get("bits") == bits]
        x = np.array([r.get("feat_rel_mse_all", np.nan) for r in sub], float)
        y = np.array([r.get("dAP", np.nan) for r in sub], float)
        m = np.isfinite(x) & np.isfinite(y)
        report["correlations"][f"bits{bits}__rel_mse_all__dAP"] = corr_block(x[m], y[m])

    # slices: per swin stage of the group
    for st in range(4):
        sub = [r for r in rows if r.get("group") and r["group"].startswith(f"L{st}")]
        x = np.array([r.get("feat_rel_mse_all", np.nan) for r in sub], float)
        y = np.array([r.get("dAP", np.nan) for r in sub], float)
        m = np.isfinite(x) & np.isfinite(y)
        report["correlations"][f"L{st}__rel_mse_all__dAP"] = corr_block(x[m], y[m])

    # matched-distortion pairs
    n = len(rows)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d1, d2 = rows[i].get("feat_rel_mse_all"), rows[j].get("feat_rel_mse_all")
            if d1 is None or d2 is None or max(d1, d2) <= 0:
                continue
            rel = abs(d1 - d2) / max(d1, d2)
            if rel < 0.10:
                dap_diff = abs(rows[i]["dAP"] - rows[j]["dAP"])
                pairs.append({
                    "i": rows[i]["tag"], "j": rows[j]["tag"],
                    "mse_i": d1, "mse_j": d2, "dap_i": rows[i]["dAP"], "dap_j": rows[j]["dAP"],
                    "dap_diff": dap_diff, "rel_mse_gap": rel,
                })
    pairs.sort(key=lambda p: -p["dap_diff"])
    report["matched_pairs"] = pairs[:50]
    report["n_matched_pairs"] = len(pairs)
    report["n_matched_pairs_dap_gt2"] = sum(1 for p in pairs if p["dap_diff"] > 2.0)

    # rank inversions (sensitivity ordering): pairs with clear MSE order but opposite ΔAP order
    inv = []
    for i in range(n):
        for j in range(i + 1, n):
            d1, d2 = rows[i].get("feat_rel_mse_all"), rows[j].get("feat_rel_mse_all")
            if d1 is None or d2 is None:
                continue
            rel = abs(d1 - d2) / max(d1, d2, 1e-12)
            if rel < 0.05:
                continue  # only consider clearly different MSE
            dap1, dap2 = rows[i]["dAP"], rows[j]["dAP"]
            if abs(dap1 - dap2) < 0.5:
                continue  # ignore tiny AP diffs (probe noise)
            if (d1 - d2) * (dap1 - dap2) < 0:
                inv.append({"i": rows[i]["tag"], "j": rows[j]["tag"],
                            "mse_i": d1, "mse_j": d2, "dap_i": dap1, "dap_j": dap2,
                            "gap": abs(dap1 - dap2)})
    inv.sort(key=lambda p: -p["gap"])
    report["rank_inversions"] = inv[:50]
    report["n_rank_inversions"] = len(inv)

    (out_dir / "correlation_report.json").write_text(json.dumps(report, indent=2))

    # ---------- figure ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    layers = sorted({int(re.match(r"L(\d)", r["group"]).group(1)) for r in rows if r.get("group")})
    for ax, xk, title in [
        (axes[0], "feat_rel_mse_all", "D_feat (all stages) vs ΔAP"),
        (axes[1], "feat_rel_mse_stage2", "D_feat (stage2) vs ΔAP"),
        (axes[2], "feat_rel_mse_stage3", "D_feat (stage3) vs ΔAP"),
    ]:
        for r in rows:
            li = int(re.match(r"L(\d)", r["group"]).group(1))
            mk = "o" if r.get("bits") == 6 else "s"
            ax.scatter(r.get(xk, np.nan), r["dAP"], color=colors[li % 4], marker=mk, s=36, alpha=0.85)
        ax.set_xlabel(xk.replace("feat_", ""))
        ax.set_ylabel("ΔAP (probe512 bbox)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    # annotations: top-5 matched pairs
    for p in pairs[:5]:
        ra = next(r for r in rows if r["tag"] == p["i"])
        axes[0].annotate("", xy=(ra["feat_rel_mse_all"], ra["dAP"]), xytext=(0, 0),
                         textcoords="offset points")
        axes[0].annotate(p["i"] + " vs " + p["j"], (ra["feat_rel_mse_all"], ra["dAP"]),
                         fontsize=6, xytext=(3, 3), textcoords="offset points")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=colors[i], marker="o", ls="", label=f"Swin stage L{i}") for i in layers]
    handles += [Line2D([0], [0], color="gray", marker="o", ls="", label="b=6"),
                Line2D([0], [0], color="gray", marker="s", ls="", label="b=4")]
    axes[0].legend(handles=handles, fontsize=7)
    fig.tight_layout()
    fig_path = ROOT / "figures/exp04_scatter.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    print("figure:", fig_path)

    # console summary
    key = report["correlations"]
    print("=== KEY CORRELATIONS (D_feat_all vs ΔAP) ===")
    print(json.dumps(key.get("feat_rel_mse_all__dAP"), indent=2))
    print("=== per bit ===")
    print(json.dumps({k: v for k, v in key.items() if k.startswith("bits")}, indent=2))
    print("=== per stage-slice ===")
    print(json.dumps({k: v for k, v in key.items() if re.match(r"L\d__", k)}, indent=2))
    print(f"matched pairs: {report['n_matched_pairs']}, with ΔAP>2: {report['n_matched_pairs_dap_gt2']}")
    print(f"rank inversions (ΔAP>0.5): {report['n_rank_inversions']}")
    print("Exp04 DONE")


if __name__ == "__main__":
    main()
