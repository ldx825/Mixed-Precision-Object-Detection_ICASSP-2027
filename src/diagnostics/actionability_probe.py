"""Quick Actionability probe on existing Exp03v2 detections (no new inference, no GT).

Legal (no-GT) proxy candidates from FP-teacher vs quantized detections on probe512:
  lost_frac  : fraction of FP detections (score>=0.3) with no same-class IoU>=0.5 match
  new_frac   : fraction of quantized detections with no FP match
  score_drop : mean score decrease on matched pairs
  iou_drop   : 1 - mean IoU on matched pairs

Question: does any legal output-space signal predict task damage better than the
F-LBQ feature-MSE proxy? (plan §31-33 Actionability pre-check)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")


def load_dets(p):
    d = json.loads(Path(p).read_text())
    by_img = {}
    for r in d:
        by_img.setdefault(r["image_id"], []).append(r)
    return by_img


def iou_mat(a, b):
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :]), 0, None)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1)
    ub = (bx2 - bx1) * (by2 - by1)
    return inter / (ua[:, None] + ub[None, :] - inter + 1e-9)


def churn(ref_by_img, q_by_img, score_thr=0.3, iou_thr=0.5):
    tot_fp = tot_matched = tot_q = 0
    score_drops, ious = [], []
    for img, refs in ref_by_img.items():
        qs = q_by_img.get(img, [])
        classes = {r["category_id"] for r in refs} | {r["category_id"] for r in qs}
        for cls in classes:
            f = [r for r in refs if r["category_id"] == cls and r["score"] >= score_thr]
            q = [r for r in qs if r["category_id"] == cls]
            tot_fp += len(f)
            tot_q += len(q)
            if not f or not q:
                continue
            fb = np.array([r["bbox"] for r in f])
            qb = np.array([r["bbox"] for r in q])
            fs = np.array([r["score"] for r in f])
            qsc = np.array([r["score"] for r in q])
            M = iou_mat(fb, qb)
            used = np.zeros(len(q), bool)
            for i in np.argsort(-fs):
                cand = np.where(~used)[0]
                if len(cand) == 0:
                    break
                j = cand[np.argmax(M[i, cand])]
                if M[i, j] >= iou_thr:
                    used[j] = True
                    tot_matched += 1
                    score_drops.append(fs[i] - qsc[j])
                    ious.append(M[i, j])
    return {
        "lost_frac": 1 - tot_matched / max(tot_fp, 1),
        "new_frac": 1 - tot_matched / max(tot_q, 1),
        "score_drop": float(np.mean(score_drops)) if score_drops else 0.0,
        "iou_drop": 1 - float(np.mean(ious)) if ious else 0.0,
        "n_fp": tot_fp, "n_q": tot_q,
    }


def corr(xs, ys):
    from scipy import stats

    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 3:
        return None
    return stats.spearmanr(xs[m], ys[m])


def main():
    fp = load_dets(ROOT / "outputs/Exp03/FP/results_bbox.json")
    runs = {}
    for d in sorted((ROOT / "outputs/Exp03v2").iterdir()):
        if d.is_dir() and (d / "results_bbox.json").exists():
            runs[d.name] = load_dets(d / "results_bbox.json")
    print("runs with detections:", len(runs))

    summary = json.loads((ROOT / "tables/Exp03v2_all_summary.json").read_text())
    AP = {"FP": summary.get("AP_FP_probe512"), "BASE4": summary.get("AP_base4")}
    D = {}
    for design, blob in summary["designs"].items():
        for r in blob["rows"]:
            AP[r["tag"]] = r["AP"]
            D[r["tag"]] = r["D"]

    proxies = {}
    for tag, dets in runs.items():
        proxies[tag] = churn(fp, dets)
        p = proxies[tag]
        print(f"[{tag:14s}] lost={p['lost_frac']:.3f} new={p['new_frac']:.3f} "
              f"score_drop={p['score_drop']:+.3f} iou_drop={p['iou_drop']:.3f} AP={AP.get(tag, float('nan')):.4f}")

    print("\n=== Actionability: legal proxy vs damage (higher = better predictor) ===")
    for design, patt in [("stage", lambda t: t.startswith("STG")),
                         ("block-only", lambda t: t.startswith("BLK") and "__b" in t)]:
        tags = [t for t in D if patt(t) and t in proxies]
        dmg = [(AP["FP"] - AP[t]) * 100 for t in tags]
        print(f"\n-- {design} (n={len(tags)}) --")
        c0 = corr([D[t] for t in tags], dmg)
        print(f"  {'D_feat(F-LBQ)':13s} spearman={c0.statistic:+.3f} p={c0.pvalue:.3f}")
        for pk in ["lost_frac", "new_frac", "score_drop", "iou_drop"]:
            c = corr([proxies[t][pk] for t in tags], dmg)
            print(f"  {pk:13s} spearman={c.statistic:+.3f} p={c.pvalue:.3f}")

    tags = [t for t in D if t.startswith("UP_") and t in proxies]
    vals = [AP[t] for t in tags]
    print(f"\n-- leave-up (n={len(tags)}), y=AP_up (higher=better config) --")
    c0 = corr([D[t] for t in tags], vals)
    print(f"  {'D_feat(F-LBQ)':13s} spearman={c0.statistic:+.3f} p={c0.pvalue:.3f}")
    for pk in ["lost_frac", "new_frac", "score_drop", "iou_drop"]:
        c = corr([proxies[t][pk] for t in tags], vals)
        print(f"  {pk:13s} spearman={c.statistic:+.3f} p={c.pvalue:.3f}")

    base = runs.get("BASE4")
    print(f"\n-- leave-up vs BASE4 output-churn (n={len(tags)}) --")
    for pk in ["lost_frac", "new_frac", "score_drop", "iou_drop"]:
        xs = [churn(base, runs[t])[pk] for t in tags]
        c = corr(xs, vals)
        print(f"  {pk:13s} spearman={c.statistic:+.3f} p={c.pvalue:.3f}")

    (ROOT / "tables/actionability_probe.json").write_text(json.dumps({"proxies": proxies, "AP": AP}, indent=2))
    print("\nwritten tables/actionability_probe.json")


if __name__ == "__main__":
    main()
