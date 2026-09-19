"""Exp34 — Calibration-size / seed robustness (Addendum §19).

For each pool seed s in 0..4 (256 imgs, nested prefixes):
  Run FP + U6 base + 12 UP configs over all 256 images;
  per-image record: n_ref (>=0.3), matched count, sum(score diffs).
  -> prefix-recomputable: score_drop(n) = sum_diff[:n] / matched[:n]

Analysis (offline, same script):
  for n in {8,16,32,64,128,256}:
    Delta_i(n, seed) = score_drop_BASE(n) - score_drop_UP_i(n)   (repair gain, positive=better)
    - per-seed ranking vs full-256 ranking (Kendall/Spearman)
    - top-4 allocation: Jaccard vs reference top-4 (256 all-seed pooled)
    - cross-seed mean Kendall of the 12-block Delta vector
Outputs: outputs/Exp34_seed_curve/{perimage_seedS.json, summary.json, n_curve.csv}
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
sys.path.insert(0, str(ROOT / "src"))

from baseline.exp01_repro_fp import build_model  # noqa: E402
from diagnostics.sweep_v2 import get_block_and_stage_groups  # noqa: E402
from quant.fake_quant import GroupQuantApplier, quantize_weight_tensor  # noqa: E402

OUT = ROOT / "outputs/Exp34_seed_curve"
OUT.mkdir(parents=True, exist_ok=True)
NS = [8, 16, 32, 64, 128, 256]


def iou_mat(a, b):
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :]), 0, None)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1)
    ub = (bx2 - bx1) * (by2 - by1)
    return inter / (ua[:, None] + ub[None, :] - inter + 1e-9)


def per_image_churn(ref_by_img, q_by_img, n_images, score_thr=0.3, iou_thr=0.5):
    """Per-image (n_ref, matched, sum_score_diff) following churn()'s matching exactly."""
    nr = np.zeros(n_images, int)
    mt = np.zeros(n_images, int)
    sd = np.zeros(n_images, float)
    for img in range(n_images):
        refs = ref_by_img.get(img, [])
        qs = q_by_img.get(img, [])
        classes = {r["category_id"] for r in refs} | {r["category_id"] for r in qs}
        for cls in classes:
            f = [r for r in refs if r["category_id"] == cls and r["score"] >= score_thr]
            q = [r for r in qs if r["category_id"] == cls]
            nr[img] += len(f)
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
                    mt[img] += 1
                    sd[img] += fs[i] - qsc[j]
    return nr, mt, sd


def run_config(model, cat_ids, n_images):
    from mmdet.apis import inference_detector

    jpgs = sorted((ROOT / "current_pool").glob("calib_*.jpg"))[:n_images]
    by = {}
    import torch

    with torch.no_grad():
        for i, p in enumerate(jpgs):
            img = np.asarray(Image.open(p).convert("RGB"))
            r = inference_detector(model, img)
            inst = r.pred_instances
            b = inst.bboxes.cpu().numpy()
            s = inst.scores.cpu().numpy()
            lb = inst.labels.cpu().numpy()
            for j in range(len(s)):
                x1, y1, x2, y2 = b[j].tolist()
                by.setdefault(i, []).append({
                    "image_id": i, "category_id": int(cat_ids[int(lb[j])]),
                    "score": float(s[j]),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                })
    return by


def main():
    model, cfg = build_model()
    applier = GroupQuantApplier(model)
    blocks, stg = get_block_and_stage_groups(model)
    pd = dict(model.named_parameters())
    all_names = {**blocks, **stg}
    blocks_sorted = sorted(blocks)

    def set_state(bits_map):
        applier.restore()
        for gname, bits in bits_map.items():
            for n in all_names[gname]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        torch.cuda.synchronize()

    from pycocotools.coco import COCO

    coco = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
    cat_ids = coco.getCatIds()

    import shutil

    for seed in range(5):
        pool_src = ROOT / f"data/calib_seeds/seed{seed}"
        pool_dst = ROOT / "current_pool"
        if pool_dst.exists():
            shutil.rmtree(pool_dst)
        shutil.copytree(pool_src, pool_dst)
        res = {}
        t0 = time.time()
        # FP teacher
        applier.restore()
        fp = run_config(model, cat_ids, 256)
        # U6 base
        set_state({g: 6 for g in blocks})
        base = run_config(model, cat_ids, 256)
        # 12 UP configs
        ups = {}
        for g in blocks_sorted:
            set_state({**{gg: 6 for gg in blocks}, g: 8})
            ups[g] = run_config(model, cat_ids, 256)
        # per-image churn
        ref_rec = per_image_churn(fp, fp, 256)
        rec = {"BASE": per_image_churn(fp, base, 256)}
        for g in blocks_sorted:
            rec[g] = per_image_churn(fp, ups[g], 256)
        ser = {k: {kk: v.tolist() for kk, v in zip(("nr", "mt", "sd"), v)}
               for k, v in rec.items()}
        (OUT / f"perimage_seed{seed}.json").write_text(json.dumps(ser))
        print(f"[seed{seed}] done in {time.time()-t0:.0f}s", flush=True)
    applier.restore()

    # ---------------- offline analysis ----------------
    from scipy import stats
    import itertools

    per_seed = {}
    for seed in range(5):
        ser = json.loads((OUT / f"perimage_seed{seed}.json").read_text())
        per_seed[seed] = {k: (np.array(v["nr"]), np.array(v["mt"]), np.array(v["sd"])) for k, v in ser.items()}

    def sd_n(seed, key, n):
        nr, mt, sd = per_seed[seed][key]
        m = mt[:n].sum()
        return float(sd[:n].sum() / m) if m > 0 else np.nan

    summary = {"n_curve": {}, "config": {"score_thr": 0.3, "iou_thr": 0.5, "base": "U6", "note": "pools from 5864-row HF train2017 subset; nested prefixes"}}
    for n in NS:
        deltas = {}
        for seed in range(5):
            d = {}
            b = sd_n(seed, "BASE", n)
            for g in blocks_sorted:
                d[g] = b - sd_n(seed, g, n)
            deltas[seed] = d
        # cross-seed agreement
        taus = []
        for s1, s2 in itertools.combinations(range(5), 2):
            v1 = np.array([deltas[s1][g] for g in blocks_sorted])
            v2 = np.array([deltas[s2][g] for g in blocks_sorted])
            taus.append(stats.kendalltau(v1, v2).statistic)
        # full-256 pooled reference top4
        ref = {}
        for g in blocks_sorted:
            b256 = np.mean([sd_n(s, "BASE", 256) for s in range(5)])
            ref[g] = b256 - np.mean([sd_n(s, g, 256) for s in range(5)])
        ref_top4 = set(sorted(blocks_sorted, key=lambda g: -ref[g])[:4])
        jacc = []
        top1_match = []
        kend_full = []
        for seed in range(5):
            top4 = set(sorted(blocks_sorted, key=lambda g: -deltas[seed][g])[:4])
            jacc.append(len(top4 & ref_top4) / 4)
            t1 = max(blocks_sorted, key=lambda g: deltas[seed][g])
            top1_match.append(t1 == max(blocks_sorted, key=lambda g: ref[g]))
            kend_full.append(stats.kendalltau(
                [deltas[seed][g] for g in blocks_sorted], [ref[g] for g in blocks_sorted]).statistic)
        summary["n_curve"][str(n)] = {
            "cross_seed_kendall_mean": float(np.mean(taus)),
            "deltas_by_seed": {str(s): deltas[s] for s in range(5)},
            "jaccard_top4_vs_full_mean": float(np.mean(jacc)),
            "top1_match_frac": float(np.mean(top1_match)),
            "kendall_vs_full_mean": float(np.mean(kend_full)),
        }
        print(f"n={n:3d}: cross-seed Kendall={np.mean(taus):+.3f}  top4-Jaccard={np.mean(jacc):.2f}  "
              f"top1-match={np.mean(top1_match):.1f}  Kendall(vs full)={np.mean(kend_full):+.3f}", flush=True)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(OUT / "n_curve.csv", "w") as f:
        f.write("n,cross_seed_kendall,top4_jaccard,top1_match,kendall_vs_full\n")
        for n in NS:
            s = summary["n_curve"][str(n)]
            f.write(f"{n},{s['cross_seed_kendall_mean']:.4f},{s['jaccard_top4_vs_full_mean']:.4f},"
                    f"{s['top1_match_frac']:.4f},{s['kendall_vs_full_mean']:.4f}\n")
    print("EXP34 DONE")


if __name__ == "__main__":
    main()
