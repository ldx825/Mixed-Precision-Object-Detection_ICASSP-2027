"""exp57 — 4.0-bit budget with 3-bit extension (bit levels {3..8}).

At the 4.0 budget every method tied at 17.41 AP (uniform-4) because all existing
methods were confined to levels {4..8}, leaving uniform as the only feasible shape.
This experiment opens level 3 and asks whether a non-uniform allocation beats
uniform-4 at the same parameter-weighted budget:

  Variant A: full-budget greedy from all-3 base (single +1 upgrades, argmax gain/cost,
             stop when no upgrade fits) -> 5k
  Variant B: reallocation from all-4 base: pair moves (i:+1, j:-1) with bits[j] > 3,
             kept within avg <= 4.0; acceptance = epsilon-tolerant split-half
             (dA+dB > 0 and min(dA,dB) >= -1e-3), up to 4 rounds -> 5k
  Variant B': the same reallocation pass starting from A's output -> 5k (if different)

Outputs: outputs/GateB/power_refined2/4.0_3bit_{A,B}.json
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
TARGET = 4.0
BIT_LO, BIT_HI = 3, 8
EPS = 1e-3


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

    def eval5k(b, tag):
        set_state(b)
        res, det_t = run_probe_detection(model, loader5k)
        m = eval_probe(res, all_ids, tag="bbox", verbose=False)
        print(f"[{tag}] 5k: AP={m['AP']:.4f} AP50={m['AP50']:.4f} avg={avg(b):.4f}", flush=True)
        return m

    # ---------- Variant A: greedy from all-3 ----------
    t0 = time.time()
    b = {g: BIT_LO for g in blocks_sorted}
    uA, uB = U_split(b)
    uS = uA + uB
    steps = 0
    while True:
        rem = TARGET - avg(b)
        best = None
        for g in blocks_sorted:
            if b[g] >= BIT_HI:
                continue
            if counts[g] / total > rem + 1e-9:
                continue
            bb = dict(b)
            bb[g] += 1
            cA, cB = U_split(bb)
            ratio = ((cA + cB) - uS) / counts[g]
            if best is None or ratio > best[2]:
                best = (g, (cA, cB), ratio)
        if best is None:
            break
        g, (cA, cB), ratio = best
        b[g] += 1
        uA, uB = cA, cB
        uS = uA + uB
        steps += 1
        print(f"  [A r{steps}] {g}->{b[g]} ratio={ratio:.3e} avg={avg(b):.4f} ({time.time()-t0:.0f}s)", flush=True)
    m = eval5k(b, "4.0-3bit-A")
    recA = {"variant": "A-greedy-from-3", "bits": b, "avg_bit": avg(b),
            "u_full": float(uS), "AP": m["AP"], "AP50": m["AP50"], "steps": steps}
    (OUT / "4.0_3bit_A.json").write_text(json.dumps(recA, indent=2))
    bits_A = dict(b)

    # ---------- Variant B: reallocation from all-4 ----------
    def reallocate(bits, tag, max_rounds=4):
        uA, uB = U_split(bits)
        t1 = time.time()
        for r in range(1, max_rounds + 1):
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
                print(f"  [{tag} r{r}] fixed point", flush=True)
                break
            (i, j, cA, cB), score = best
            bits[i] += 1
            bits[j] -= 1
            uA, uB = cA, cB
            print(f"  [{tag} r{r}] {i}+1/{j}-1 gain={score:+.5f} avg={avg(bits):.4f} ({time.time()-t1:.0f}s)", flush=True)
        return bits, uA + uB

    b4 = {g: 4 for g in blocks_sorted}
    b4, uB_full = reallocate(b4, "B-from-4")
    mB = eval5k(b4, "4.0-3bit-B")
    recB = {"variant": "B-realloc-from-4", "bits": b4, "avg_bit": avg(b4),
            "u_full": float(uB_full), "AP": mB["AP"], "AP50": mB["AP50"]}
    (OUT / "4.0_3bit_B.json").write_text(json.dumps(recB, indent=2))

    # ---------- Variant B': reallocation starting from A ----------
    bA2 = dict(bits_A)
    bA2, uA2_full = reallocate(bA2, "B'-from-A", max_rounds=2)
    if bA2 != bits_A:
        mA2 = eval5k(bA2, "4.0-3bit-Bprime")
        recBp = {"variant": "Bprime-realloc-from-A", "bits": bA2, "avg_bit": avg(bA2),
                 "u_full": float(uA2_full), "AP": mA2["AP"], "AP50": mA2["AP50"]}
        (OUT / "4.0_3bit_Bprime.json").write_text(json.dumps(recBp, indent=2))
    else:
        print("[B'] no change from A; skipped 5k", flush=True)

    applier.restore()
    print("EXP57 DONE", flush=True)


if __name__ == "__main__":
    main()
