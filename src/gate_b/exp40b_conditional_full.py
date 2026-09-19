"""exp40b — FULL-budget conditional greedy for fixed targets 4.5 and 5.5.

Fixes the prefix-truncation flaw of exp40 snapshots (which stopped at the greedy path's
first step that overshot the target, leaving budget unspent).
Here: fresh trajectory from U4 for each target; stop only when NO single +1-level upgrade
fits the remaining budget. Outputs: outputs/GateB/cond_full_{45,55}.json (+ trajectory files).
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

OUT = ROOT / "outputs/GateB"
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = [4.5, 5.5]


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

    def set_state(bvec):
        applier.restore()
        for g, bits in bvec.items():
            for n in all_names[g]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        torch.cuda.synchronize()

    applier.restore()
    fp_dets = run_calib_dets(model, cat_ids)

    def U_of(bvec):
        set_state(bvec)
        return -churn(fp_dets, run_calib_dets(model, cat_ids))["score_drop"]

    def avg(b):
        return sum(counts[g] * b[g] for g in blocks_sorted) / total

    for T in TARGETS:
        out_file = OUT / f"cond_full_{int(T*10)}.json"
        t0 = time.time()
        b = {g: 4 for g in blocks_sorted}
        uS = U_of(b)
        steps = []
        while True:
            rem = T - avg(b)
            best = None
            for g in blocks_sorted:
                if b[g] >= 8:
                    continue
                if counts[g] / total > rem + 1e-9:  # one more level must fit
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
            (OUT / f"cond_full_{int(T*10)}_traj.json").write_text(json.dumps(
                {"target": T, "current_bits": {gg: b[gg] for gg in blocks_sorted}, "steps": steps}, indent=2))
            print(f"[T={T}] {g}->{b[g]} ratio={ratio:.3e} avg={avg(b):.4f} ({time.time()-t0:.0f}s)", flush=True)
        res = {f"{T:.1f}": {"conditional-OUT-full": {"bits": b, "avg_bit": avg(b), "u": float(uS)}}}
        out_file.write_text(json.dumps(res, indent=2))
        print(f"[T={T}] FINAL avg={avg(b):.4f} bits={b}", flush=True)
    applier.restore()
    print("EXP40B DONE", flush=True)


if __name__ == "__main__":
    main()
