"""exp61 — Swin-S reference sweep: uniform-5, uniform-6, conditional@6.0.

Completes the Swin-S picture so the conditional allocation can be compared at
matched and adjacent budgets. Reuses the exp60 loading/eval path.

Output: outputs/GateB_swins/{uniform5,uniform6,cond60}_5k.json
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
from diagnostics.confirm_5k import run_calib_dets  # noqa: E402
from diagnostics.actionability_probe import churn  # noqa: E402
from diagnostics.sweep_groupbits import build_probe_loader, run_probe_detection  # noqa: E402
from utils.coco_eval_subset import eval_probe  # noqa: E402

OUT = ROOT / "outputs/GateB_swins"
OUT.mkdir(parents=True, exist_ok=True)
TARGET = 6.0

CKPT = ROOT / "checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
CFG_CANDS = [
    ROOT / "third_party/mmdetection/configs/swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py",
]


def main():
    cfg_path = next((c for c in CFG_CANDS if Path(c).exists()), None)
    model, cfg = build_model(cfg_path=str(cfg_path), ckpt=str(CKPT))
    applier = GroupQuantApplier(model)
    blocks, stg = get_block_and_stage_groups(model)
    pd = dict(model.named_parameters())
    all_names = {**blocks, **stg}
    blocks_sorted = sorted(blocks)
    counts = {g: sum(pd[n].numel() for n in blocks[g]) for g in blocks_sorted}
    total = sum(counts.values())

    from pycocotools.coco import COCO

    coco = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
    cat_ids = coco.getCatIds()
    all_ids = sorted(int(i) for i in coco.getImgIds())
    loader5k = build_probe_loader(cfg, all_ids, workers=6)

    def set_state(bvec):
        applier.restore()
        for g, bits in bvec.items():
            for n in all_names[g]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        torch.cuda.synchronize()

    def avg(b):
        return sum(counts[g] * b[g] for g in blocks_sorted) / total

    def eval5k(b, tag, out_name):
        set_state(b)
        res, det_t = run_probe_detection(model, loader5k)
        m = eval_probe(res, all_ids, tag="bbox", verbose=False)
        print(f"[{tag}] 5k: AP={m['AP']:.4f} AP50={m['AP50']:.4f} avg={avg(b):.4f}", flush=True)
        (OUT / out_name).write_text(json.dumps(
            {"bits": b, "avg_bit": avg(b), "AP": m["AP"], "AP50": m["AP50"]}, indent=2))
        return m

    applier.restore()
    fp_dets = run_calib_dets(model, cat_ids)

    def U_of(bvec):
        set_state(bvec)
        return -churn(fp_dets, run_calib_dets(model, cat_ids))["score_drop"]

    # ---- uniform-5 and uniform-6 ----
    b5 = {g: 5 for g in blocks_sorted}
    eval5k(b5, "swins-uniform5", "uniform5_5k.json")
    b6 = {g: 6 for g in blocks_sorted}
    eval5k(b6, "swins-uniform6", "uniform6_5k.json")

    # ---- conditional full-budget greedy @6.0 ----
    t0 = time.time()
    b = {g: 4 for g in blocks_sorted}
    uS = U_of(b)
    steps = []
    while True:
        rem = TARGET - avg(b)
        best = None
        for g in blocks_sorted:
            if b[g] >= 8:
                continue
            if counts[g] / total > rem + 1e-9:
                continue
            bb = dict(b)
            bb[g] += 1
            u = U_of(bb)
            ratio = (u - uS) / counts[g]
            if best is None or ratio > best[2]:
                best = (g, u, ratio)
        if best is None:
            break
        g, u, ratio = best
        b[g] += 1
        uS = u
        steps.append({"g": g, "to_bits": b[g], "u": float(u), "ratio": float(ratio)})
        print(f"[swins-greedy60] {g}->{b[g]} ratio={ratio:.3e} avg={avg(b):.4f} ({time.time()-t0:.0f}s)", flush=True)
    set_state(b)
    res, det_t = run_probe_detection(model, loader5k)
    m = eval_probe(res, all_ids, tag="bbox", verbose=False)
    print(f"[swins-cond60] 5k: AP={m['AP']:.4f} AP50={m['AP50']:.4f} avg={avg(b):.4f}", flush=True)
    (OUT / "cond60_5k.json").write_text(json.dumps(
        {"bits": b, "avg_bit": avg(b), "u": float(uS), "AP": m["AP"], "AP50": m["AP50"],
         "steps": steps}, indent=2))

    applier.restore()
    print("EXP61 DONE", flush=True)


if __name__ == "__main__":
    main()
