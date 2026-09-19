"""exp42 — deployable swap-refinement for conditional allocations + 5k evaluation.

Takes an allocation file (e.g. cond_full_45.json), swap-refines each entry on calib64
utility (1-level +/-1 changes, budget-constrained, <=3 rounds; GT-free), then 5k-evaluates
the refined allocation. Writes outputs/GateB/refined_results.json (independent of eval_summary).

usage: python exp42_refine.py outputs/GateB/cond_full_45.json
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
MAX_ROUNDS = 3


def main():
    alloc_file = Path(sys.argv[1])
    data = json.loads(alloc_file.read_text())

    model, cfg = build_model()
    applier = GroupQuantApplier(model)
    blocks, stg = get_block_and_stage_groups(model)
    pd = dict(model.named_parameters())
    all_names = {**blocks, **stg}
    blocks_sorted = sorted(blocks)
    counts = {g: sum(pd[n].numel() for n in blocks[g]) for g in blocks_sorted}
    total = sum(counts.values())

    def set_state(bvec):
        applier.restore()
        for g, bits in bvec.items():
            for n in all_names[g]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        torch.cuda.synchronize()

    def avg(b):
        return sum(counts[g] * b[g] for g in blocks_sorted) / total

    from pycocotools.coco import COCO

    coco = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
    cat_ids = coco.getCatIds()
    all_ids = sorted(int(i) for i in coco.getImgIds())
    loader = build_probe_loader(cfg, all_ids, workers=6)

    applier.restore()
    fp_dets = run_calib_dets(model, cat_ids)

    def U_of(bvec):
        set_state(bvec)
        return -churn(fp_dets, run_calib_dets(model, cat_ids))["score_drop"]

    results = {}
    t0 = time.time()
    for tag, entry in data.items():
        target = float(tag)
        bits = {g: int(v) for g, v in entry.get("bits", entry.get("conditional-OUT-full", {}).get("bits", {})).items()}
        if not bits:
            print(f"[{tag}] no bits found, skipping"); continue
        u = U_of(bits)
        print(f"[{tag}] start U={u:+.5f} avg={avg(bits):.4f}", flush=True)
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
                    if avg(cand) > target + 1e-9:
                        continue
                    u2 = U_of(cand)
                    if u2 > u + 1e-12 and (best is None or u2 > best[1]):
                        best = ((g, d), u2)
            if best is None:
                break
            (g, d), u2 = best
            bits[g] += d
            u = u2
            print(f"  [{tag} r{rounds}] swap {g}{d:+d} -> U={u:+.5f} avg={avg(bits):.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
        # 5k evaluation
        set_state(bits)
        res, det_t = run_probe_detection(model, loader)
        m = eval_probe(res, all_ids, tag="bbox", verbose=False)
        print(f"[{tag}] REFINED 5k: AP={m['AP']:.4f} AP50={m['AP50']:.4f} ({det_t:.0f}s)", flush=True)
        results[tag] = {"bits": bits, "avg_bit": avg(bits), "u": float(u), "AP": m["AP"],
                        "AP50": m["AP50"], "rounds": rounds}
        (OUT / f"refined_{alloc_file.stem}_results.json").write_text(json.dumps(results, indent=2))
    applier.restore()
    print("EXP42 DONE", flush=True)
    for tag, v in results.items():
        print(f"  {tag}: AP={v['AP']:.4f} avg={v['avg_bit']:.3f} bits={v['bits']}")


if __name__ == "__main__":
    main()
