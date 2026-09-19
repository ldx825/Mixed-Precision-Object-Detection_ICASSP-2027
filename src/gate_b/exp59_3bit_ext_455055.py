"""exp59 — level-3 extension for 4.5/5.0/5.5 budgets (from current-best starts).

Same reallocation pass as exp57 variant B (pair moves i:+1/j:-1 with bits[j] > 3,
kept within avg <= target; epsilon-tolerant split-half acceptance
dA+dB > 0 and min(dA,dB) >= -1e-3), but starting from each budget's current
best allocation. Up to 4 rounds per budget, then 5k evaluation.

Motivation: extending the level set to include 3-bit gave +2.08 AP at the
4.0 budget (exp57); this tests the same extension at 4.5/5.0/5.5.

Starts: power_refined2/4.5_pair.json, 5.0_pair.json, 5.5_pair.json (fallback:
power_refined/<tag>_power*.json).
Output: outputs/GateB/power_refined2/{tag}_3bit.json
"""
import json
import sys
import time
from itertools import permutations
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
TARGETS = ["4.5", "5.0", "5.5"]
BIT_LO, BIT_HI = 3, 8
EPS = 1e-3
MAX_ROUNDS = 4

STARTS = {
    "4.5": ["outputs/GateB/power_refined2/4.5_pair.json"],
    "5.0": ["outputs/GateB/power_refined2/5.0_pair.json"],
    "5.5": ["outputs/GateB/power_refined2/5.5_pair.json",
            "outputs/GateB/power_refined/5.5_power_refined_cond_full_55_results.json"],
}


def load_any_bits(path, tag):
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text())
    if tag in data and isinstance(data[tag], dict):
        entry = data[tag]
        if "bits" in entry:
            return {g: int(v) for g, v in entry["bits"].items()}
        for v in entry.values():
            if isinstance(v, dict) and "bits" in v:
                return {g: int(x) for g, x in v["bits"].items()}
    if "bits" in data:
        return {g: int(v) for g, v in data["bits"].items()}
    for v in data.values():
        if isinstance(v, dict):
            if "bits" in v:
                return {g: int(x) for g, x in v["bits"].items()}
            for v2 in v.values():
                if isinstance(v2, dict) and "bits" in v2:
                    return {g: int(x) for g, x in v2["bits"].items()}
    return None


def main():
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

    for tag in TARGETS:
        T = float(tag)
        bits = None
        for path in STARTS[tag]:
            bits = load_any_bits(path, tag)
            if bits is not None:
                print(f"[{tag}] start from {path}", flush=True)
                break
        if bits is None:
            print(f"[{tag}] NO START FOUND; skipping", flush=True)
            continue
        t0 = time.time()
        uA, uB = U_split(bits)
        print(f"[{tag}] start avg={avg(bits):.4f} uA={uA:+.5f} uB={uB:+.5f}", flush=True)
        for r in range(1, MAX_ROUNDS + 1):
            best = None
            for i, j in permutations(blocks_sorted, 2):
                if bits[i] >= BIT_HI or bits[j] <= BIT_LO:
                    continue
                cand = dict(bits)
                cand[i] += 1
                cand[j] -= 1
                if avg(cand) > T + 1e-9:
                    continue
                cA, cB = U_split(cand)
                dA, dB = cA - uA, cB - uB
                if dA + dB > 0 and min(dA, dB) >= -EPS:
                    score = dA + dB
                    if best is None or score > best[1]:
                        best = ((i, j, cA, cB), score)
            if best is None:
                print(f"  [{tag} r{r}] fixed point", flush=True)
                break
            (i, j, cA, cB), score = best
            bits[i] += 1
            bits[j] -= 1
            uA, uB = cA, cB
            print(f"  [{tag} r{r}] {i}+1/{j}-1 gain={score:+.5f} avg={avg(bits):.4f} ({time.time()-t0:.0f}s)", flush=True)
        set_state(bits)
        res, det_t = run_probe_detection(model, loader5k)
        m = eval_probe(res, all_ids, tag="bbox", verbose=False)
        print(f"[{tag}] 3BIT-EXT 5k: AP={m['AP']:.4f} AP50={m['AP50']:.4f} avg={avg(bits):.4f}", flush=True)
        rec = {"bits": bits, "avg_bit": avg(bits), "uA": float(uA), "uB": float(uB),
               "u_full": float(uA + uB), "AP": m["AP"], "AP50": m["AP50"],
               "variant": "level3-realloc-from-best"}
        (OUT / f"{tag}_3bit.json").write_text(json.dumps(rec, indent=2))
    applier.restore()
    print("EXP59 DONE", flush=True)


if __name__ == "__main__":
    main()
