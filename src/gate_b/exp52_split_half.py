"""exp52 — split-half validated refinement (XL) — 64-image contract, final sprint.

Accept a swap only if it improves BOTH halves of the 64-image calibration set (first-32 /
last-32; split-half cross-validation). This filters swaps that are artifacts of calibration
sampling noise (seed-noise ~0.0016 at n=64, comparable to typical swap effects).
No data beyond the 64-image contract is used.

Multi-start (ours / strongest-prior / oracle allocations), <=4 rounds each (fixed point);
final 5k bbox AP. Output: outputs/GateB/power_refined2/{tag}_xl.json
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

OUT = ROOT / "outputs/GateB/power_refined2"
OUT.mkdir(parents=True, exist_ok=True)

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
        if "bits" in entry:
            return {g: int(v) for g, v in entry["bits"].items()}
        for v in entry.values():
            if isinstance(v, dict) and "bits" in v:
                return {g: int(x) for g, x in v["bits"].items()}
        raise ValueError(f"no bits in {spec_path}[{tag}]")
    return {g: int(v) for g, v in entry[method]["bits"].items()}


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
    fp64 = run_calib_dets(model, cat_ids)
    fpA = {k: v for k, v in fp64.items() if k < 32}
    fpB = {k: v for k, v in fp64.items() if k >= 32}

    def U_split(b):
        set_state(b)
        dets = run_calib_dets(model, cat_ids)
        A = {k: v for k, v in dets.items() if k < 32}
        B = {k: v for k, v in dets.items() if k >= 32}
        return -churn(fpA, A)["score_drop"], -churn(fpB, B)["score_drop"]

    t0 = time.time()
    results = {}
    for name, spec_path, sub in STARTS[tag]:
        bits = load_bits(spec_path, sub)
        uA, uB = U_split(bits)
        print(f"[{tag}/{name}] start avg={avg(bits):.4f} uA={uA:+.5f} uB={uB:+.5f} ({time.time()-t0:.0f}s)", flush=True)
        for r in range(1, 5):
            best = None
            for g in blocks_sorted:
                for d in (+1, -1):
                    cand = dict(bits)
                    cand[g] += d
                    if cand[g] < 4 or cand[g] > 8:
                        continue
                    if avg(cand) > target + 1e-9:
                        continue
                    cA, cB = U_split(cand)
                    dA = cA - uA
                    dB = cB - uB
                    if dA > 0 and dB > 0:  # split-half validated: both halves improve
                        score = min(dA, dB)
                        if best is None or score > best[1]:
                            best = ((g, d, cA, cB), score)
            if best is None:
                print(f"  [{tag}/{name} r{r}] no split-half-validated swap (fixed point)", flush=True)
                break
            (g, d, cA, cB), score = best
            bits[g] += d
            uA, uB = cA, cB
            print(f"  [{tag}/{name} r{r}] swap {g}{d:+d} min-gain={score:+.5f} avg={avg(bits):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        results[name] = {"bits": bits, "avg_bit": avg(bits), "uA": float(uA), "uB": float(uB),
                         "u_full": float(uA + uB)}

    best_name = max(results, key=lambda k: results[k]["u_full"])
    bits = results[best_name]["bits"]
    print(f"[{tag}] best start: {best_name} u_full={results[best_name]['u_full']:+.5f}", flush=True)

    set_state(bits)
    res, det_t = run_probe_detection(model, loader5k)
    m = eval_probe(res, all_ids, tag="bbox", verbose=False)
    print(f"[{tag}] XL FINAL 5k: AP={m['AP']:.4f} AP50={m['AP50']:.4f}", flush=True)
    rec = {"best_start": best_name, "bits": bits, "avg_bit": avg(bits),
           "uA": results[best_name]["uA"], "uB": results[best_name]["uB"],
           "AP": m["AP"], "AP50": m["AP50"],
           "all_starts": {k: {"avg_bit": v["avg_bit"], "u_full": v["u_full"]} for k, v in results.items()}}
    (OUT / f"{tag}_xl.json").write_text(json.dumps(rec, indent=2))
    applier.restore()
    print(f"EXP52 DONE [{tag}] AP={m['AP']:.4f} bits={bits}", flush=True)


if __name__ == "__main__":
    main()
