"""exp56 — FULL-budget conditional greedy for targets 5.0 and 6.0 (exp40b extension).

exp40b (the fixed full-budget procedure) covered 4.5/5.5 only; 5.0/6.0 used the
older truncated exp40 snapshots (the 6.0 allocation stopped at avg=5.9487).
This runs the fixed procedure directly at 5.0 and 6.0:
  single +1 upgrades, argmax gain-per-cost, stop only when no upgrade fits.
Then evaluates each output on COCO 5k.

Outputs: outputs/GateB/cond_full_50.json, cond_full_60.json (bits + u + 5k AP)
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

OUT = ROOT / "outputs/GateB"
TARGETS = [5.0, 6.0]


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
            print(f"[T={T}] {g}->{b[g]} ratio={ratio:.3e} avg={avg(b):.4f} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[T={T}] greedy done avg={avg(b):.4f} u={uS:+.5f}; evaluating 5k...", flush=True)
        set_state(b)
        res, det_t = run_probe_detection(model, loader5k)
        m = eval_probe(res, all_ids, tag="bbox", verbose=False)
        print(f"[T={T}] FULL-GREEDY 5k: AP={m['AP']:.4f} AP50={m['AP50']:.4f}", flush=True)
        out = {f"{T:.1f}": {"conditional-full": {"bits": b, "avg_bit": avg(b),
               "u": float(uS), "AP": m["AP"], "AP50": m["AP50"], "steps": steps}}}
        out_file.write_text(json.dumps(out, indent=2))
    applier.restore()
    print("EXP56 DONE", flush=True)


if __name__ == "__main__":
    main()
