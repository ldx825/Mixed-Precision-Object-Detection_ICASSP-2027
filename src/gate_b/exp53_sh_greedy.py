"""exp53 — split-half-validated greedy allocation (method v2) — 64-img contract.

METHOD CHANGE (vs exp40): every greedy upgrade must improve BOTH halves of the 64-image
calibration set (split-half cross-validation) — systematic noise suppression in the
allocation search itself, not just in refinement. Then split-half refine (<=3 rounds), 5k.

Output: outputs/GateB/power_refined2/{tag}_shgreedy.json
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
MAX_STEPS = 60
MAX_REFINE_ROUNDS = 3


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
    bits = {g: 4 for g in blocks_sorted}
    uA, uB = U_split(bits)
    print(f"[{tag}] SH-greedy start avg=4.0 uA={uA:+.5f} uB={uB:+.5f}", flush=True)

    # ---- split-half-validated greedy upgrades ----
    for step in range(1, MAX_STEPS + 1):
        best = None
        for g in blocks_sorted:
            if bits[g] >= 8:
                continue
            cand = dict(bits)
            cand[g] += 1
            if avg(cand) > target + 1e-9:
                continue
            cA, cB = U_split(cand)
            dA, dB = cA - uA, cB - uB
            if dA > 0 and dB > 0:
                score = min(dA, dB)
                if best is None or score > best[1]:
                    best = ((g, cA, cB), score)
        if best is None:
            print(f"[{tag}] no more split-half-validated upgrades at step {step}", flush=True)
            break
        (g, cA, cB), score = best
        bits[g] += 1
        uA, uB = cA, cB
        print(f"[{tag}] step {step}: {g}->{bits[g]} min-gain={score:+.5f} avg={avg(bits):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- split-half refine ----
    for r in range(1, MAX_REFINE_ROUNDS + 1):
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
                dA, dB = cA - uA, cB - uB
                if dA > 0 and dB > 0:
                    score = min(dA, dB)
                    if best is None or score > best[1]:
                        best = ((g, d, cA, cB), score)
        if best is None:
            print(f"[{tag}] refine fixed point at round {r}", flush=True)
            break
        (g, d, cA, cB), score = best
        bits[g] += d
        uA, uB = cA, cB
        print(f"[{tag}] refine r{r}: {g}{d:+d} min-gain={score:+.5f} avg={avg(bits):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 5k ----
    set_state(bits)
    res, det_t = run_probe_detection(model, loader5k)
    m = eval_probe(res, all_ids, tag="bbox", verbose=False)
    print(f"[{tag}] SH-GREEDY FINAL 5k: AP={m['AP']:.4f} AP50={m['AP50']:.4f}", flush=True)
    rec = {"bits": bits, "avg_bit": avg(bits), "uA": float(uA), "uB": float(uB),
           "AP": m["AP"], "AP50": m["AP50"]}
    (OUT / f"{tag}_shgreedy.json").write_text(json.dumps(rec, indent=2))
    applier.restore()
    print(f"EXP53 DONE [{tag}] AP={m['AP']:.4f} bits={bits}", flush=True)


if __name__ == "__main__":
    main()
