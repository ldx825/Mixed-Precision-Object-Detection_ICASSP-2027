"""Analyze Exp03v2 leave-up design: D_feat(repair) vs AP benefit + rank comparison.

Benefit(g) = AP_BASE4 - AP_UP(g)  (positive: repairing block g recovers AP)
D_feat(g)  = ||F_up(g) - F_base||^2 / ||F_up(g)||^2  (calib64, mean)

If feature-alteration is a good proxy for detection damage, D_feat(g) should
rank-correlate with Benefit(g). Weak/negative correlation or heavy rank
inversions => proxy mismatch (Case C candidate).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
sys.path.insert(0, str(ROOT / "src"))

from utils.coco_eval_subset import eval_probe  # noqa: E402


def main():
    d = ROOT / "outputs/Exp03v2"
    tables = ROOT / "tables"
    tables.mkdir(exist_ok=True)

    rows = []
    for run in sorted(d.iterdir()):
        if not run.is_dir():
            continue
        rf = run / "results_bbox.json"
        if not rf.exists():
            continue
        res = json.loads(rf.read_text())
        ids = sorted({r["image_id"] for r in res})
        m = eval_probe(res, ids, tag="bbox")
        fd = json.loads((run / "feat_dist.json").read_text()) if (run / "feat_dist.json").exists() else {}
        meta = json.loads((run / "run_meta.json").read_text()) if (run / "run_meta.json").exists() else {}
        rows.append({
            "tag": run.name, "group": meta.get("group"),
            **{f"AP_{k}": v for k, v in m.items()},
            **{f"feat_{k}": v for k, v in fd.items() if isinstance(v, (int, float))},
        })
    print(f"runs evaluated: {len(rows)}")

    base = next((r for r in rows if r["tag"] == "BASE4"), None)
    if base is None:
        print("BASE4 missing — cannot compute benefit")
        return
    for r in rows:
        r["benefit_AP"] = (base["AP_AP"] - r["AP_AP"]) * 100  # AP points
        for k in ["AP50", "AP75", "AP_S", "AP_M", "AP_L"]:
            r[f"benefit_{k}"] = (base[f"AP_{k}"] - r[f"AP_{k}"]) * 100

    ups = [r for r in rows if r["tag"].startswith("UP_")]
    print(f"UP runs: {len(ups)}; BASE4 AP={base['AP_AP']:.4f}")

    from scipy import stats

    corr_out = {}
    x = np.array([r.get("feat_rel_mse_all", np.nan) for r in ups], float)
    for yk in ["benefit_AP", "benefit_AP50", "benefit_AP75", "benefit_AP_S", "benefit_AP_M", "benefit_AP_L"]:
        y = np.array([r.get(yk, np.nan) for r in ups], float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 2:
            sp = stats.spearmanr(x[m], y[m])
            kt = stats.kendalltau(x[m], y[m])
            pr = stats.pearsonr(x[m], y[m])
            corr_out[yk] = {"spearman": float(sp.statistic), "spearman_p": float(sp.pvalue),
                            "kendall": float(kt.statistic), "kendall_p": float(kt.pvalue),
                            "pearson": float(pr.statistic), "n": int(m.sum())}
            print(f"{yk:14s} spearman={sp.statistic:+.3f} (p={sp.pvalue:.3f})  kendall={kt.statistic:+.3f}  pearson={pr.statistic:+.3f}")

    # rank comparison table
    print(f"\n{'block':10s} {'benefit_AP':>10s} {'D_feat':>10s} {'AP':>7s}  {'rank_b':>6s} {'rank_d':>6s}")
    by_benefit = sorted(ups, key=lambda r: -r.get("benefit_AP", 0))
    by_feat = sorted(ups, key=lambda r: -r.get("feat_rel_mse_all", 0))
    rb = {r["tag"]: i + 1 for i, r in enumerate(by_benefit)}
    rd = {r["tag"]: i + 1 for i, r in enumerate(by_feat)}
    for r in by_benefit:
        print(f"{r['group']:10s} {r.get('benefit_AP',0):+10.3f} {r.get('feat_rel_mse_all',0):10.6f} "
              f"{r.get('AP_AP',0):7.4f}  {rb[r['tag']]:6d} {rd[r['tag']]:6d}")

    # inversions: pairs with clear feats order opposite to benefit order
    inv = 0
    pairs = []
    n = len(ups)
    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = ups[i].get("feat_rel_mse_all", 0), ups[j].get("feat_rel_mse_all", 0)
            bi, bj = ups[i].get("benefit_AP", 0), ups[j].get("benefit_AP", 0)
            if max(fi, fj) <= 0:
                continue
            if abs(fi - fj) / max(fi, fj) < 0.05:
                continue
            if abs(bi - bj) < 0.2:
                continue
            if (fi - fj) * (bi - bj) < 0:
                inv += 1
                pairs.append({"i": ups[i]["tag"], "j": ups[j]["tag"],
                              "feat_i": fi, "feat_j": fj, "ben_i": bi, "ben_j": bj,
                              "gap": abs(bi - bj)})
    pairs.sort(key=lambda p: -p["gap"])
    print(f"\nrank inversions (clear feat gap, benefit gap>0.2): {inv} of {n*(n-1)//2} pairs")

    out = {"base": base, "rows": rows, "correlations": corr_out,
           "rank_inversions": pairs[:30], "n_inversions": inv}
    (tables / "Exp03v2_summary.json").write_text(json.dumps(out, indent=2))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for r in ups:
        xx = r.get("feat_rel_mse_all", 0)
        yy = r.get("benefit_AP", 0)
        ax.scatter(xx, yy, s=60, color="#1f77b4")
        ax.annotate(r["group"], (xx, yy), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("D_feat (repair-induced rel MSE on calib64)")
    ax.set_ylabel("AP benefit of repairing block (points)")
    ax.set_title("Exp03v2 leave-up: feature alteration vs detection benefit (Swin-T MRCNN, probe512)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figp = ROOT / "figures/exp03v2_leaveup.png"
    figp.parent.mkdir(exist_ok=True)
    fig.savefig(figp, dpi=160)
    print("written:", figp)
    print("written:", tables / "Exp03v2_summary.json")


if __name__ == "__main__":
    main()
