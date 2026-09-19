"""Gate B — conditional greedy allocation from U4 base (Addendum §15) + static-OUT capture.

One greedy trajectory (max budget 6.0, single-block +1-level upgrades, ratio (U gain)/n_g).
Prefix truncations give the allocations for lower budgets (hard budget satisfied at every step).
Also captures the first-step signals:
  static_out_signal.json : Delta(g: 4->8 direct repair) per block on U4 base (calib64).
Resumable via outputs/GateB/cond_traj.json.
Output: outputs/GateB/cond_traj.json (+ {budget: bits} in cond_alloc.json at the end)
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
MAX_BUDGET = 6.0
BUDGETS = [6.0, 5.5, 5.0, 4.5, 4.0]


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
        ch = churn(fp_dets, run_calib_dets(model, cat_ids))
        return -ch["score_drop"], ch

    def spent(bvec):
        return sum(counts[g] * (bvec[g] - 4) for g in blocks_sorted)

    traj_file = OUT / "cond_traj.json"
    if traj_file.exists():
        traj = json.loads(traj_file.read_text())
        b = {g: int(v) for g, v in traj["current_bits"].items()}
        steps = traj["steps"]
        print(f"[resume] {len(steps)} steps done, spent={spent(b)}", flush=True)
    else:
        b = {g: 4 for g in blocks_sorted}
        steps = []

    t0 = time.time()
    uS, chS = U_of(b)
    print(f"[start] U4 U={uS:+.5f}", flush=True)

    # ---- first-step static-OUT signal (4 -> 8 direct) ----
    so_file = OUT / "static_out_signal.json"
    if not so_file.exists() and len(steps) == 0:
        sig = {}
        for g in blocks_sorted:
            bb = dict(b)
            bb[g] = 8
            u, ch = U_of(bb)
            sig[g] = {"delta_4to8": float(u - uS), "score_drop_8": float(ch["score_drop"])}
            print(f"  [SO {g}] Delta(4->8)={u-uS:+.5f}", flush=True)
        so_file.write_text(json.dumps(sig, indent=2))
        print("[static-OUT signal saved]", flush=True)

    # ---- greedy loop up to MAX_BUDGET ----
    R_max = (MAX_BUDGET - 4.0) * total
    while True:
        rem = R_max - spent(b)
        best = None
        for g in blocks_sorted:
            if b[g] >= 8:
                continue
            if counts[g] > rem + 1e-6:
                continue
            bb = dict(b)
            bb[g] += 1
            u, ch = U_of(bb)
            ratio = (u - uS) / counts[g]
            if best is None or ratio > best[2]:
                best = (g, u, ratio, ch)
        if best is None:
            print("[stop] no feasible/beneficial upgrade", flush=True)
            break
        g, u, ratio, ch = best
        b[g] += 1
        uS = u
        steps.append({"g": g, "to_bits": b[g], "u": float(u), "ratio": float(ratio),
                      "spent_after": int(spent(b)),
                      "snapshot_bits": {gg: b[gg] for gg in blocks_sorted}})
        (OUT / "cond_traj.json").write_text(json.dumps(
            {"current_bits": {g: b[g] for g in blocks_sorted}, "steps": steps}, indent=2))
        print(f"[step {len(steps)}] {g} -> {b[g]}bit  u={u:+.5f} ratio={ratio:.3e} "
              f"spent={spent(b)/total:.3f}bit avg ({time.time()-t0:.0f}s)", flush=True)
        if spent(b) >= R_max - 1e-6:
            break

    # ---- budget snapshots (longest prefix with spent <= (B-4)*total) ----
    snaps = {}
    for B in BUDGETS:
        R_B = (B - 4.0) * total
        best_bits = {g: 4 for g in blocks_sorted}
        for st in steps:
            if st["spent_after"] <= R_B + 1e-6:
                best_bits = st["snapshot_bits"]
            else:
                break
        avg = sum(counts[g] * best_bits[g] for g in blocks_sorted) / total
        snaps[f"{B:.1f}"] = {"bits": best_bits, "avg_bit": avg}
    (OUT / "cond_alloc.json").write_text(json.dumps(snaps, indent=2))
    applier.restore()
    print("COND DONE", flush=True)
    for B, v in snaps.items():
        print(f"  B={B}: avg={v['avg_bit']:.3f}  bits={v['bits']}")


if __name__ == "__main__":
    main()
