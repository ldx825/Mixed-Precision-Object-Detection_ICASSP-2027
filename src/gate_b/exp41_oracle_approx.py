"""Gate B — oracle-approx: 1-swap local search over feasible single-block +/-1 changes,
starting from the conditional-greedy allocations (cond_alloc.json). calib64 utility (GT-free).
Output: outputs/GateB/allocations_oracle.json  {budget: {"oracle-approx": {bits, avg_bit, u}}}
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
BUDGETS = ["6.0", "5.5", "5.0", "4.5", "4.0"]
MAX_ROUNDS = 3


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

    alloc = json.loads((OUT / "cond_alloc.json").read_text())
    res = {}
    t0 = time.time()
    for B in BUDGETS:
        bits = {g: int(v) for g, v in alloc[B]["bits"].items()}
        u = U_of(bits)
        print(f"[B={B}] start U={u:+.5f} avg={avg(bits):.3f}", flush=True)
        rounds = 0
        while rounds < MAX_ROUNDS:
            rounds += 1
            best = None
            for g in blocks_sorted:
                for d in (+1, -1):
                    cand = dict(bits)
                    cand[g] += d
                    if cand[g] < 4 or cand[g] > 8:
                        continue
                    if avg(cand) > float(B) + 1e-9:
                        continue
                    u2 = U_of(cand)
                    if u2 > u + 1e-12 and (best is None or u2 > best[1]):
                        best = ((g, d), u2)
            if best is None:
                break
            (g, d), u2 = best
            bits[g] += d
            u = u2
            print(f"  [B={B} r{rounds}] swap {g}{d:+d} -> U={u:+.5f} "
                  f"avg={avg(bits):.3f} ({time.time()-t0:.0f}s)", flush=True)
        res[B] = {"oracle-approx": {"bits": bits, "avg_bit": avg(bits), "u": float(u)}}
    (OUT / "allocations_oracle.json").write_text(json.dumps(res, indent=2))
    applier.restore()
    print("ORACLE-APPROX DONE", flush=True)
    for B in BUDGETS:
        v = res[B]["oracle-approx"]
        print(f"  B={B}: U={v['u']:+.5f} avg={v['avg_bit']:.3f} bits={v['bits']}")


if __name__ == "__main__":
    main()
