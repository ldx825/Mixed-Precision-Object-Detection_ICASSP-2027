"""exp51 — FINAL sprint: multi-start + 320-image ensemble-verification refinement.

For each budget: refine from 3 starts (our best / F-LBQ-style best / oracle-approx),
Stage-B-style swap on a 320-image ensemble (calib_seeds seed0..4 x 64 each, to suppress
calibration sampling noise), 2 rounds per start; pick best by U320; final 5k eval.
All GT-free. Output: outputs/GateB/power_refined2/{tag}_final.json
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
from diagnostics.actionability_probe import churn  # noqa: E402
from diagnostics.sweep_groupbits import build_probe_loader, run_probe_detection  # noqa: E402
from utils.coco_eval_subset import eval_probe  # noqa: E402

OUT = ROOT / "outputs/GateB/power_refined2"
OUT.mkdir(parents=True, exist_ok=True)
POOLS = [(ROOT / f"data/calib_seeds/seed{s}", 64) for s in range(5)]  # 320 imgs total

STARTS = {
    "4.5": [
        ("ours", "outputs/GateB/power_refined/4.5_power.json", None),
        ("flbq", "outputs/GateB/allocations_static.json", ("4.5", "F-LBQ-reimpl")),
        ("oracle", "outputs/GateB/allocations_oracle.json", ("4.5", "oracle-approx")),
    ],
    "5.5": [
        ("ours", "outputs/GateB/power_refined/5.5_power_refined_cond_full_55_results.json", None),
        ("staticout", "outputs/GateB/allocations_static.json", ("5.5", "static-OUT")),
        ("ours_old", "outputs/GateB/cond_alloc.json", ("5.5", None)),
    ],
    "5.0": [
        ("ours", "outputs/GateB/power_refined/5.0_power_allocations_oracle.json", None),
        ("blobq", "outputs/GateB/allocations_static.json", ("5.0", "BLOBQ-reimpl-greedy")),
        ("oracle", "outputs/GateB/allocations_oracle.json", ("5.0", "oracle-approx")),
    ],
}


def run_multi_pool_dets(model, cat_ids):
    from mmdet.apis import inference_detector

    by = {}
    gidx = 0
    with torch.no_grad():
        for pool_dir, n in POOLS:
            files = sorted(p for p in pool_dir.iterdir() if p.suffix in (".jpg", ".png"))[:n]
            for p in files:
                img = np.asarray(Image.open(p).convert("RGB"))
                r = inference_detector(model, img)
                inst = r.pred_instances
                b = inst.bboxes.cpu().numpy()
                s = inst.scores.cpu().numpy()
                lb = inst.labels.cpu().numpy()
                for j in range(len(s)):
                    x1, y1, x2, y2 = b[j].tolist()
                    by.setdefault(gidx, []).append({
                        "image_id": gidx, "category_id": int(cat_ids[int(lb[j])]),
                        "score": float(s[j]),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    })
                gidx += 1
    return by


def load_bits(spec_path, sub):
    data = json.loads((ROOT / spec_path).read_text())
    if sub is None:
        entry = data
        if "bits" in entry:
            return {g: int(v) for g, v in entry["bits"].items()}
        for v in entry.values():
            if isinstance(v, dict):
                if "bits" in v:
                    return {g: int(v) for g, v in v["bits"].items()}
                for v2 in v.values():
                    if isinstance(v2, dict) and "bits" in v2:
                        return {g: int(x) for g, x in v2["bits"].items()}
    tag, method = sub
    entry = data[tag]
    if method is None:
        entry = entry.get("bits", next((v for v in entry.values() if isinstance(v, dict)), entry))
    else:
        entry = entry[method]
    return {g: int(v) for g, v in entry["bits"].items()}


def main():
    tag = sys.argv[1]
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
    fp320 = run_multi_pool_dets(model, cat_ids)

    def U320(b):
        set_state(b)
        return -churn(fp320, run_multi_pool_dets(model, cat_ids))["score_drop"]

    t0 = time.time()
    results = {}
    for name, spec_path, sub in STARTS[tag]:
        bits = load_bits(spec_path, sub)
        u = U320(bits)
        print(f"[{tag}/{name}] start avg={avg(bits):.4f} u320={u:+.5f} ({time.time()-t0:.0f}s)", flush=True)
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
                    u2 = U320(cand)
                    if u2 > u + 1e-12 and (best is None or u2 > best[1]):
                        best = ((g, d), u2)
            if best is None:
                break
            (g, d), u = best
            bits[g] += d
            print(f"  [{tag}/{name} r{r}] swap {g}{d:+d} u320={u:+.5f} avg={avg(bits):.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
        results[name] = {"bits": bits, "avg_bit": avg(bits), "u320": float(u)}

    # pick best by U320
    best_name = max(results, key=lambda k: results[k]["u320"])
    bits = results[best_name]["bits"]
    print(f"[{tag}] best start: {best_name} u320={results[best_name]['u320']:+.5f}", flush=True)

    set_state(bits)
    res, det_t = run_probe_detection(model, loader5k)
    m = eval_probe(res, all_ids, tag="bbox", verbose=False)
    print(f"[{tag}] FINAL 5k: AP={m['AP']:.4f} AP50={m['AP50']:.4f}", flush=True)
    rec = {"best_start": best_name, "bits": bits, "avg_bit": avg(bits),
           "u320": results[best_name]["u320"], "AP": m["AP"], "AP50": m["AP50"],
           "all_starts": {k: {"avg_bit": v["avg_bit"], "u320": v["u320"]} for k, v in results.items()}}
    (OUT / f"{tag}_final.json").write_text(json.dumps(rec, indent=2))
    applier.restore()
    print(f"EXP51 DONE [{tag}] AP={m['AP']:.4f} bits={bits}", flush=True)


if __name__ == "__main__":
    main()
