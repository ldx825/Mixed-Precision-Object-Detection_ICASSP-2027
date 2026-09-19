"""5k re-evaluation of 12 decisive Exp03v2 points (plan §20 Stage 2).

Resolves probe512 noise on the strongest signal points:
  leave-up: BASE4 + UP_BLK{2.3,3.0,2.5,3.1,0.0,0.1,2.0}
  stage:    STG{2,0,3,1}__b4
Writes outputs/Exp03v2_5k/<tag>/{results_bbox.json,done.json} + metrics_5k.json
and prints refreshed leave-up correlations.
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
sys.path.insert(0, str(ROOT / "src"))

from baseline.exp01_repro_fp import build_model  # noqa: E402
from diagnostics.sweep_v2 import get_block_and_stage_groups  # noqa: E402
from quant.fake_quant import GroupQuantApplier, quantize_weight_tensor  # noqa: E402
from diagnostics.sweep_groupbits import build_probe_loader, run_probe_detection  # noqa: E402
from utils.coco_eval_subset import eval_probe  # noqa: E402

POINTS = [
    "BASE4",
    "UP_BLK2.3", "UP_BLK3.0", "UP_BLK2.5", "UP_BLK3.1",
    "UP_BLK0.0", "UP_BLK0.1", "UP_BLK2.0",
    "STG2__b4", "STG0__b4", "STG3__b4", "STG1__b4",
]


def bits_map_for(tag, blocks, stg):
    if tag == "BASE4":
        return {g: 4 for g in blocks}
    if tag.startswith("UP_"):
        g = tag[3:]
        return {**{gg: 4 for gg in blocks}, g: 8}
    if "__b" in tag:
        s, b = tag.split("__b")
        return {s: int(b)}
    raise ValueError(tag)


def main():
    out_root = ROOT / "outputs/Exp03v2_5k"
    out_root.mkdir(exist_ok=True)

    model, cfg = build_model()
    applier = GroupQuantApplier(model)
    blocks, stg = get_block_and_stage_groups(model)
    pd = dict(model.named_parameters())
    all_names = {**blocks, **stg}

    from pycocotools.coco import COCO

    coco = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
    all_ids = sorted(int(i) for i in coco.getImgIds())
    print(f"val images: {len(all_ids)}", flush=True)
    loader = build_probe_loader(cfg, all_ids, workers=6)

    def set_state(bits_map):
        applier.restore()
        for gname, bits in bits_map.items():
            for n in all_names[gname]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        torch.cuda.synchronize()

    metrics = {}
    for tag in POINTS:
        d = out_root / tag
        if not (d / "done.json").exists():
            d.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            set_state(bits_map_for(tag, blocks, stg))
            res, det_t = run_probe_detection(model, loader)
            (d / "results_bbox.json").write_text(json.dumps(res))
            (d / "done.json").write_text(json.dumps({"n": len(res), "det_s": det_t}))
            print(f"[{tag}] {len(res)} dets, {det_t:.0f}s", flush=True)
        else:
            print(f"[skip-infer] {tag}", flush=True)
        res = json.loads((d / "results_bbox.json").read_text())
        m = eval_probe(res, sorted({r["image_id"] for r in res}), tag="bbox", verbose=False)
        metrics[tag] = m
        print(f"[{tag}] AP={m['AP']:.4f} AP50={m['AP50']:.4f} AP_S={m['AP_S']:.4f} AP_L={m['AP_L']:.4f}", flush=True)
        (out_root / "metrics_5k.json").write_text(json.dumps(metrics, indent=2))

    applier.restore()

    # refreshed correlations
    import numpy as np
    from scipy import stats

    up = [t for t in POINTS if t.startswith("UP_")]
    D, Y = [], []
    for t in up:
        fd = json.loads((ROOT / "outputs/Exp03v2" / t / "feat_dist.json").read_text())
        D.append(fd["rel_mse_all"])
        Y.append(metrics[t]["AP"])
    sp = stats.spearmanr(D, Y)
    kt = stats.kendalltau(D, Y)
    print(f"\n[5k leave-up] Spearman(D, AP_up)={sp.statistic:+.3f} (p={sp.pvalue:.4f}) "
          f"Kendall={kt.statistic:+.3f} n={len(D)}", flush=True)
    stg_tags = [t for t in POINTS if t.startswith("STG")]
    Ds, Ys = [], []
    for t in stg_tags:
        fd = json.loads((ROOT / "outputs/Exp03v2" / t / "feat_dist.json").read_text())
        Ds.append(fd["rel_mse_all"])
        Ys.append(metrics["BASE4"]["AP"] * 0 + 1)  # placeholder unused
    # stage damage correlation
    fp_ap = None
    fp_res = json.loads((ROOT / "outputs/Exp03/FP/results_bbox.json").read_text())
    fp_ids = sorted({r["image_id"] for r in fp_res})
    # NOTE: FP probe512 file only; recompute FP on 5k from Exp01 fixed results instead
    fp5k = json.loads((ROOT / "outputs/Exp01/results_bbox_fixed.json").read_text())
    mfp = eval_probe(fp5k, sorted({r["image_id"] for r in fp5k}), tag="bbox", verbose=False)
    print(f"[FP 5k] AP={mfp['AP']:.4f}", flush=True)
    Ds, Ys = [], []
    for t in stg_tags:
        fd = json.loads((ROOT / "outputs/Exp03v2" / t / "feat_dist.json").read_text())
        Ds.append(fd["rel_mse_all"])
        Ys.append((mfp["AP"] - metrics[t]["AP"]) * 100)
    if len(Ds) >= 3:
        sp2 = stats.spearmanr(Ds, Ys)
        print(f"[5k stage] Spearman(D, damage)={sp2.statistic:+.3f} (p={sp2.pvalue:.4f}) n={len(Ds)}", flush=True)

    (out_root / "metrics_5k.json").write_text(json.dumps({"points": metrics, "fp_5k": mfp}, indent=2))
    print("5K EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
