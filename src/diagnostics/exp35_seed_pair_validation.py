"""Exp35 — cross-seed validation of pair interactions + |S|=1 rank reversal.

Uses pools data/calib_seeds/seed{1..4} (256 imgs; first-64 prefix matches Exp34 slices).
For each seed: FP, U6 base, 12 single-block UP (6->8), 66 pairs (both 6->8).
Analysis: per-seed I_cent stats; cross-seed correlation of the 66 interaction values;
per-seed |S|=1 reversal rate vs singleton ranking; Delta (UP curbs) cross-seed agreement.
Outputs: outputs/Exp35_seed_pairs/seed{s}_runs.json, summary.json
"""
import itertools
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
from diagnostics.actionability_probe import churn  # noqa: E402

OUT = ROOT / "outputs/Exp35_seed_pairs"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [1, 2, 3, 4]
NIMG = 64


def run_pool_dets(model, cat_ids, pool_dir, n=NIMG):
    from mmdet.apis import inference_detector

    jpgs = sorted(pool_dir.glob("calib_*.jpg"))[:n]
    by = {}
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
    pairs = list(itertools.combinations(blocks_sorted, 2))

    from pycocotools.coco import COCO

    coco = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
    cat_ids = coco.getCatIds()

    def set_state(bits_map):
        applier.restore()
        for gname, bits in bits_map.items():
            for n in all_names[gname]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        torch.cuda.synchronize()

    for seed in SEEDS:
        run_file = OUT / f"seed{seed}_runs.json"
        runs = json.loads(run_file.read_text()) if run_file.exists() else {}
        # heal int keys after json round-trip (churn matches by image index)
        if "FP" in runs and runs["FP"] and isinstance(next(iter(runs["FP"])), str):
            runs["FP"] = {int(k): v for k, v in runs["FP"].items()}
        pool = ROOT / f"data/calib_seeds/seed{seed}"

        applier.restore()
        if "FP" not in runs:
            runs["FP"] = run_pool_dets(model, cat_ids, pool)
            run_file.write_text(json.dumps(runs))
            print(f"[seed{seed}] FP: {len(runs['FP'])} imgs with dets", flush=True)

        def run_cfg(key, bits_map):
            if key in runs:
                return
            set_state(bits_map)
            dets = run_pool_dets(model, cat_ids, pool)
            ch = churn(runs["FP"], dets)
            runs[key] = {"churn": ch}
            run_file.write_text(json.dumps(runs))
            print(f"[seed{seed}] {key}: score_drop={ch['score_drop']:+.5f} lost={ch['lost_frac']:.4f}", flush=True)

        run_cfg("U6", {g: 6 for g in blocks_sorted})
        for g in blocks_sorted:
            run_cfg(f"UP_{g}", {**{gg: 6 for gg in blocks_sorted}, g: 8})
        for (i, j) in pairs:
            run_cfg(f"P_{i}+{j}", {**{gg: 6 for gg in blocks_sorted}, i: 8, j: 8})
    applier.restore()

    # ---------------- analysis ----------------
    from scipy import stats

    per_seed = {}
    for seed in SEEDS:
        runs = json.loads((OUT / f"seed{seed}_runs.json").read_text())
        u0 = -runs["U6"]["churn"]["score_drop"]
        sing = {g: -runs[f"UP_{g}"]["churn"]["score_drop"] for g in blocks_sorted}
        delta = {g: sing[g] - u0 for g in blocks_sorted}
        I = {}
        for (i, j) in pairs:
            uij = -runs[f"P_{i}+{j}"]["churn"]["score_drop"]
            I[f"{i}+{j}"] = uij - sing[i] - sing[j] + u0
        # |S|=1 reversal
        revs = []
        for j in blocks_sorted:
            cond, base = {}, {}
            for i in blocks_sorted:
                if i == j:
                    continue
                key = f"{i}+{j}" if f"{i}+{j}" in runs else f"{j}+{i}"
                uij = -runs[key]["churn"]["score_drop"]
                cond[i] = uij - sing[j]
                base[i] = delta[i]
            ks = sorted(cond)
            tot = rev = 0
            for a, b in itertools.combinations(ks, 2):
                d = cond[a] - cond[b]
                e = base[a] - base[b]
                if d == 0 or e == 0:
                    continue
                tot += 1
                if (d > 0) != (e > 0):
                    rev += 1
            revs.append(rev / max(tot, 1))
        per_seed[seed] = {"delta": delta, "I": I, "reversal": float(np.mean(revs))}

    # cross-seed stats (including seed0 from Exp30)
    s0 = json.loads((ROOT / "outputs/Exp30_pair_interaction/runs.json").read_text())
    blk0 = sorted(p.name.replace("UP_", "") for p in (ROOT / "outputs/Exp04_budget6").glob("UP_*") if p.is_dir())
    sing0 = {g: -json.loads((ROOT / f"outputs/Exp04_budget6/UP_{g}/calib_churn.json").read_text())["score_drop"] for g in blk0}
    u00 = -s0["BASE"]["score_drop"]
    I0 = {}
    for (i, j) in pairs:
        I0[f"{i}+{j}"] = -s0[f"{i}+{j}"]["score_drop"] - sing0[i] - sing0[j] + u00
    per_seed[0] = {"I": I0, "delta": {g: sing0[g] - u00 for g in blk0}}

    keys = sorted(I0)
    mat = np.array([[per_seed[s]["I"][k] for k in keys] for s in sorted(per_seed)])
    corr = np.corrcoef(mat)
    kts = [[stats.kendalltau(mat[a], mat[b]).statistic for b in range(len(mat))] for a in range(len(mat))]
    summary = {
        "I_stats_per_seed": {str(s): {"mean_abs": float(np.mean(np.abs(list(per_seed[s]["I"].values())))),
                                      "ratio_vs_delta": float(np.mean(np.abs(list(per_seed[s]["I"].values()))) /
                                                              np.mean(np.abs(list(per_seed[s]["delta"].values()))))}
                             for s in sorted(per_seed)},
        "I_pearson_matrix": corr.tolist(),
        "I_kendall_matrix": kts,
        "reversal_per_seed": {str(s): per_seed[s]["reversal"] for s in sorted(per_seed) if "reversal" in per_seed[s]},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== I |mean| / mean|Delta| ratio per seed ===")
    for s in sorted(per_seed):
        st = summary["I_stats_per_seed"][str(s)]
        print(f"  seed{s}: mean|I|={st['mean_abs']:.5f}  ratio={st['ratio_vs_delta']:.2f}")
    print("\n=== pairwise Pearson corr of I-matrices (seeds 0..4) ===")
    for a in range(len(mat)):
        print("  " + " ".join(f"{corr[a][b]:+.2f}" for b in range(len(mat))))
    print("\n=== |S|=1 reversal rate per seed ===")
    for s in sorted(per_seed):
        if "reversal" in per_seed[s]:
            print(f"  seed{s}: reversal={per_seed[s]['reversal']*100:.1f}%")
    print("EXP35 DONE")


if __name__ == "__main__":
    main()
