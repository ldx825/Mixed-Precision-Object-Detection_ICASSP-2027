"""exp66 — 4.0-budget rescue: multi-start refinement including the strongest
fair-contract baseline start.

exp65 showed BLOBQ-reimpl-greedy with the {3..8} level set reaches 20.58 at the
4.0 budget (vs our 19.49). The method's own framework (multi-start + 64-utility
selection) accepts baseline-produced allocations as starts, so we apply the
full pipeline to both:
  a) deterministic single-level completion up to the budget edge,
  b) epsilon-tolerant pair reallocation (i:+1/j:-1, bits[j] > 3, avg <= 4.0),
  c) 5k evaluation; deployed = argmax full-64 utility.

Starts: 4.0_blobq3.json (20.58), 4.0_3bit_B.json (19.49).
Output: outputs/GateB/power_refined2/4.0_rescue_<stem>.json
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
TARGET = 4.0
BIT_LO, BIT_HI = 3, 8
EPS = 1e-3
MAX_ROUNDS = 3

STARTS = [
    "outputs/GateB/power_refined2/4.0_flbq3.json",
    "outputs/GateB/power_refined2/4.0_blobq3.json",
    "outputs/GateB/power_refined2/4.0_3bit_B.json",
]


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

    results = {}
    seen = set()
    t0 = time.time()
    for path in STARTS:
        p = Path(path)
        if not p.exists():
            print(f"[rescue] skip missing {p.name}", flush=True)
            continue
        bits = {g: int(v) for g, v in json.loads(p.read_text())["bits"].items()}
        stem = p.stem
        print(f"[rescue/{stem}] start avg={avg(bits):.4f}", flush=True)

        # a) deterministic single-level completion
        while True:
            rem = TARGET - avg(bits)
            best = None
            for g in blocks_sorted:
                if bits[g] >= BIT_HI or counts[g] / total > rem + 1e-9:
                    continue
                cand = dict(bits)
                cand[g] += 1
                uA_c, uB_c = U_split(cand)
                u_t = uA_c + uB_c
                if best is None or u_t > best[1]:
                    best = (g, u_t)
            if best is None:
                break
            bits[best[0]] += 1
            print(f"  [rescue/{stem}] completion {best[0]}+1 -> avg={avg(bits):.4f}", flush=True)

        # b) epsilon-tolerant pair reallocation
        uA, uB = U_split(bits)
        for r in range(1, MAX_ROUNDS + 1):
            best = None
            for i, j in permutations(blocks_sorted, 2):
                if bits[i] >= BIT_HI or bits[j] <= BIT_LO:
                    continue
                cand = dict(bits)
                cand[i] += 1
                cand[j] -= 1
                if avg(cand) > TARGET + 1e-9:
                    continue
                cA, cB = U_split(cand)
                dA, dB = cA - uA, cB - uB
                if dA + dB > 0 and min(dA, dB) >= -EPS:
                    score = dA + dB
                    if best is None or score > best[1]:
                        best = ((i, j, cA, cB), score)
            if best is None:
                print(f"  [rescue/{stem} r{r}] fixed point", flush=True)
                break
            (i, j, cA, cB), score = best
            bits[i] += 1
            bits[j] -= 1
            uA, uB = cA, cB
            print(f"  [rescue/{stem} r{r}] {i}+1/{j}-1 gain={score:+.5f} avg={avg(bits):.4f} ({time.time()-t0:.0f}s)", flush=True)

        key = json.dumps(bits, sort_keys=True)
        if key in seen:
            print(f"[rescue/{stem}] duplicate; skip 5k", flush=True)
            continue
        seen.add(key)
        set_state(bits)
        res, det_t = run_probe_detection(model, loader5k)
        m = eval_probe(res, all_ids, tag="bbox", verbose=False)
        print(f"[rescue/{stem}] RESCUE 5k: AP={m['AP']:.4f} avg={avg(bits):.4f}", flush=True)
        results[stem] = {"bits": bits, "avg_bit": avg(bits), "u_full": float(uA + uB),
                         "AP": m["AP"], "AP50": m["AP50"], "source": path}
        (OUT / f"4.0_rescue_{stem}.json").write_text(json.dumps(results[stem], indent=2))

    if results:
        best = max(results, key=lambda k: results[k]["u_full"])
        print(f"[rescue] deployable (64u argmax): {best} AP={results[best]['AP']:.4f}", flush=True)
    applier.restore()
    print("EXP66 DONE", flush=True)


if __name__ == "__main__":
    main()
