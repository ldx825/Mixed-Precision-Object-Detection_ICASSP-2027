"""exp50 — POWER refinement for Gate B allocations (all GT-free).

Stage A: single-block +/-1 swap on calib64 (deployment seed), up to 6 rounds (fixed point).
Stage B: swap refinements re-evaluated on calib256 (seed0 pool = 256 imgs), 2 rounds:
         higher-precision measurements catch swaps that 64-img noise hides.
Multi-init: optionally refine from several starting allocations, keep the best by U256.
Final: 5k bbox AP evaluation.

usage: python exp50_power_refine.py <input_json> <budget_tag>
  input_json: {"<tag>": "<method>" -> {"bits": {...}}}  (or nested like refined_* files)
Output: outputs/GateB/power_refined/<tag>_power.json  {bits, u64, u256, AP, AP50}
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
from diagnostics.confirm_5k import run_calib_dets  # noqa: E402
from diagnostics.actionability_probe import churn  # noqa: E402
from diagnostics.sweep_groupbits import build_probe_loader, run_probe_detection  # noqa: E402
from utils.coco_eval_subset import eval_probe  # noqa: E402

OUT = ROOT / "outputs/GateB/power_refined"
OUT.mkdir(parents=True, exist_ok=True)
POOL256 = ROOT / "data/calib_seeds/seed0"


def run_pool_dets(model, cat_ids, pool_dir, n):
    from mmdet.apis import inference_detector

    files = sorted(p for p in pool_dir.iterdir() if p.suffix in (".jpg", ".png"))[:n]
    by = {}
    with torch.no_grad():
        for i, p in enumerate(files):
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


def extract_bits(entry):
    if "bits" in entry:
        return {g: int(v) for g, v in entry["bits"].items()}
    for sub in entry.values():
        if isinstance(sub, dict):
            if "bits" in sub:
                return {g: int(v) for g, v in sub["bits"].items()}
            for sub2 in sub.values():
                if isinstance(sub2, dict) and "bits" in sub2:
                    return {g: int(v) for g, v in sub2["bits"].items()}
    raise ValueError("no bits found")


def main():
    inp = Path(sys.argv[1])
    tag = sys.argv[2]
    data = json.loads(inp.read_text())
    bits0 = extract_bits(data[tag])
    target = float(tag)

    model, cfg = build_model()
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

    applier.restore()
    fp64 = run_calib_dets(model, cat_ids)
    fp256 = run_pool_dets(model, cat_ids, POOL256, 256)

    def U64(b):
        set_state(b)
        return -churn(fp64, run_calib_dets(model, cat_ids))["score_drop"]

    def U256(b):
        set_state(b)
        return -churn(fp256, run_pool_dets(model, cat_ids, POOL256, 256))["score_drop"]

    t0 = time.time()
    bits = dict(bits0)
    u64 = U64(bits)
    print(f"[{tag}] start: avg={avg(bits):.4f} u64={u64:+.5f} bits={bits}", flush=True)

    # ---- Stage A: 64-img swap to fixed point (max 6 rounds) ----
    for r in range(1, 7):
        best = None
        for g in blocks_sorted:
            for d in (+1, -1):
                cand = dict(bits)
                cand[g] += d
                if cand[g] < 4 or cand[g] > 8:
                    continue
                if avg(cand) > target + 1e-9:
                    continue
                u2 = U64(cand)
                if u2 > u64 + 1e-12 and (best is None or u2 > best[1]):
                    best = ((g, d), u2)
        if best is None:
            break
        (g, d), u64 = best
        bits[g] += d
        print(f"  [A{r}] swap {g}{d:+d} u64={u64:+.5f} avg={avg(bits):.4f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- Stage B: 256-img swap (2 rounds, higher precision) ----
    u256 = U256(bits)
    print(f"  [B0] u256={u256:+.5f}", flush=True)
    for r in range(1, 3):
        best = None
        for g in blocks_sorted:
            for d in (+1, -1):
                cand = dict(bits)
                cand[g] += d
                if cand[g] < 4 or cand[g] > 8:
                    continue
                if avg(cand) > target + 1e-9:
                    continue
                u2 = U256(cand)
                if u2 > u256 + 1e-12 and (best is None or u2 > best[1]):
                    best = ((g, d), u2)
        if best is None:
            break
        (g, d), u256 = best
        bits[g] += d
        print(f"  [B{r}] swap {g}{d:+d} u256={u256:+.5f} avg={avg(bits):.4f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- final 5k ----
    set_state(bits)
    res, det_t = run_probe_detection(model, loader5k)
    m = eval_probe(res, all_ids, tag="bbox", verbose=False)
    print(f"[{tag}] POWER-REFINED 5k: AP={m['AP']:.4f} AP50={m['AP50']:.4f}", flush=True)
    rec = {"bits": bits, "avg_bit": avg(bits), "u64": float(u64), "u256": float(u256),
           "AP": m["AP"], "AP50": m["AP50"], "n_rounds_AB": None}
    (OUT / f"{tag}_power_{inp.stem}.json").write_text(json.dumps(rec, indent=2))
    applier.restore()
    print(f"EXP50 DONE [{tag}] AP={m['AP']:.4f} bits={bits}", flush=True)


if __name__ == "__main__":
    main()
