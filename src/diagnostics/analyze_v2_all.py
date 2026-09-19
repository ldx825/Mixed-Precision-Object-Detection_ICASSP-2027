"""Merged analysis across all Exp03v2 designs (leave-up / stage / block-only).

- leave-up   : x = D_feat(repair-induced vs base4); y = AP_up (repaired AP).
               Proxy-consistent prediction: larger D_feat -> lower AP_up
               (repairing a "large-alteration" block should disturb more).
- stage      : x = D_feat (quantized vs FP);        y = damage = AP_FP - AP.
- block-only : x = D_feat (quantized vs FP);        y = damage = AP_FP - AP.

For stage/block-only, a good proxy means larger D_feat -> larger damage.
Reports per-design Spearman/Kendall/Pearson, rank tables and a 3-panel figure.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
sys.path.insert(0, str(ROOT / "src"))

from utils.coco_eval_subset import eval_probe  # noqa: E402

AP_FP_PROBE512 = None  # computed from outputs/Exp03/FP


def load_rows(d):
    rows = []
    for run in sorted(d.iterdir()):
        rf = run / "results_bbox.json"
        if not run.is_dir() or not rf.exists():
            continue
        res = json.loads(rf.read_text())
        if not res:
            continue
        ids = sorted({r["image_id"] for r in res})
        m = eval_probe(res, ids, tag="bbox")
        fd = json.loads((run / "feat_dist.json").read_text()) if (run / "feat_dist.json").exists() else {}
        meta = json.loads((run / "run_meta.json").read_text()) if (run / "run_meta.json").exists() else {}
        rows.append({
            "tag": run.name, "group": meta.get("group"), "design": meta.get("design"),
            "bits": meta.get("bits"),
            "AP": m["AP"], "AP50": m["AP50"], "AP75": m["AP75"],
            "AP_S": m["AP_S"], "AP_M": m["AP_M"], "AP_L": m["AP_L"],
            "D": fd.get("rel_mse_all"),
        })
    return rows


def corr(x, y):
    from scipy import stats

    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return {}
    sp = stats.spearmanr(x[m], y[m])
    kt = stats.kendalltau(x[m], y[m])
    pr = stats.pearsonr(x[m], y[m])
    return {"spearman": float(sp.statistic), "spearman_p": float(sp.pvalue),
            "kendall": float(kt.statistic), "kendall_p": float(kt.pvalue),
            "pearson": float(pr.statistic), "n": int(m.sum())}


def main():
    global AP_FP_PROBE512
    d = ROOT / "outputs/Exp03v2"
    out = {"designs": {}}

    # FP reference on probe512
    fp_res = json.loads((ROOT / "outputs/Exp03/FP/results_bbox.json").read_text())
    fp_ids = sorted({r["image_id"] for r in fp_res})
    AP_FP_PROBE512 = eval_probe(fp_res, fp_ids, tag="bbox")["AP"]
    out["AP_FP_probe512"] = AP_FP_PROBE512
    print(f"FP probe512 AP = {AP_FP_PROBE512:.4f}")

    rows = load_rows(d)
    base4 = next((r for r in rows if r["tag"] == "BASE4"), None)
    if base4:
        out["AP_base4"] = base4["AP"]
        print(f"BASE4 probe512 AP = {base4['AP']:.4f}")

    # leave-up
    lu = [r for r in rows if r["tag"].startswith("UP_")]
    for r in lu:
        r["y_APup"] = r["AP"]
    c = corr([r["D"] for r in lu], [r["y_APup"] for r in lu])
    out["designs"]["leave-up"] = {"corr_D_vs_APup": c, "rows": lu}
    print("\n[leave-up] Spearman(D, AP_up) =", round(c.get("spearman", float("nan")), 3),
          "p=", round(c.get("spearman_p", float("nan")), 3), "n=", c.get("n"))

    # stage
    stg = [r for r in rows if r["tag"].startswith("STG")]
    for r in stg:
        r["damage"] = (AP_FP_PROBE512 - r["AP"]) * 100
    c = corr([r["D"] for r in stg], [r["damage"] for r in stg])
    out["designs"]["stage"] = {"corr_D_vs_damage": c, "rows": stg}
    print("[stage]    Spearman(D, damage) =", round(c.get("spearman", float("nan")), 3),
          "p=", round(c.get("spearman_p", float("nan")), 3), "n=", c.get("n"))

    # block-only
    blk = [r for r in rows if r["tag"].startswith("BLK") and "__b" in r["tag"]]
    for r in blk:
        r["damage"] = (AP_FP_PROBE512 - r["AP"]) * 100
    c = corr([r["D"] for r in blk], [r["damage"] for r in blk])
    out["designs"]["block-only"] = {"corr_D_vs_damage": c, "rows": blk}
    print("[block-only] Spearman(D, damage) =", round(c.get("spearman", float("nan")), 3),
          "p=", round(c.get("spearman_p", float("nan")), 3), "n=", c.get("n"))

    # rank tables
    for key, arr, ykey in [("leave-up", lu, "AP"), ("stage", stg, "damage"), ("block-only", blk, "damage")]:
        if not arr:
            continue
        print(f"\n--- {key} rank table (desc by y) ---")
        for r in sorted(arr, key=lambda r: -r[ykey]):
            print(f"  {r['tag']:16s} D={r['D']:.6f} AP={r['AP']:.4f} y={r[ykey]:+.3f}")

    (ROOT / "tables/Exp03v2_all_summary.json").write_text(json.dumps(out, indent=2))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (key, arr, ykey, ylabel) in zip(axes, [
        ("leave-up", lu, "y_APup", "AP_up (repaired)"),
        ("stage", stg, "damage", "damage (AP points)"),
        ("block-only", blk, "damage", "damage (AP points)"),
    ]):
        for r in arr:
            ax.scatter(r["D"], r[ykey], s=50)
            ax.annotate(r["tag"], (r["D"], r[ykey]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("D_feat (rel MSE)")
        ax.set_ylabel(ylabel)
        ax.set_title(key)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = ROOT / "figures/exp03v2_designs.png"
    fig.savefig(fp, dpi=160)
    print("\nwritten:", fp)
    print("written:", ROOT / "tables/Exp03v2_all_summary.json")


if __name__ == "__main__":
    main()
