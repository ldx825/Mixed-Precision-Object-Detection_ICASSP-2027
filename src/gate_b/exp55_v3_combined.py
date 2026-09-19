"""exp55 — epsilon-tolerant COMBINED swap refinement (weapon v3) — 64-img contract.

Acceptance: dA + dB > 0  AND  min(dA, dB) >= -EPS   (EPS=0.001)
  - looser than strict split-half (both halves must improve; too conservative per exp53),
  - stricter than unvalidated swaps (noise-prone per early refinements):
  allows small half-wise wobble but requires the total two-half gain to be positive.
Move set: single-block +/-1  AND  budget-feasible pair moves (i:+1, j:-1).
Multi-start (best known allocations), <=8 rounds to fixed point, 5k eval.
Output: outputs/GateB/power_refined2/{tag}_v3.json
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
EPS = 0.001
MAX_ROUNDS = 8

START_FILES = {
    "4.5": ["outputs/GateB/power_refined/4.5_power.json",
            "outputs/GateB/power_refined2/4.5_xl.json"],
    "5.5": ["outputs/GateB/power_refined/5.5_power_refined_cond_full_55_results.json",
            "outputs/GateB/cond_alloc.json"],
    "5.0": ["outputs/GateB/power_refined/5.0_power_allocations_oracle.json",
            "outputs/GateB/power_refined2/5.0_xl.json"],
    "6.0": ["outputs/GateB/cond_alloc.json",
            "outputs/GateB/allocations_oracle.json"],
}


def load_any_bits(path, tag=None):
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text())
    if tag and tag in data and isinstance(data[tag], dict):
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
    for path in START_FILES[tag]:
        bits = load_any_bits(path, tag)
        if bits is None:
            print(f"[{tag}] skip missing/unparsable {path}", flush=True)
            continue
        uA, uB = U_split(bits)
        print(f"[{tag}/{Path(path).stem}] start avg={avg(bits):.4f} uA={uA:+.5f} uB={uB:+.5f}", flush=True)
        for r in range(1, MAX_ROUNDS + 1):
            best = None
            moves = []
            for g in blocks_sorted:
                for d in (+1, -1):
                    moves.append(("single", g, d))
            for i, j in permutations(blocks_sorted, 2):
                if bits[i] < 8 and bits[j] > 4:
                    moves.append(("pair", i, j))
            for mv in moves:
                cand = dict(bits)
                if mv[0] == "single":
                    _, g, d = mv
                    cand[g] += d
                    if cand[g] < 4 or cand[g] > 8:
                        continue
                else:
                    _, i, j = mv
                    cand[i] += 1
                    cand[j] -= 1
                if avg(cand) > target + 1e-9:
                    continue
                cA, cB = U_split(cand)
                dA, dB = cA - uA, cB - uB
                if dA + dB > 0 and min(dA, dB) >= -EPS:
                    score = dA + dB
                    if best is None or score > best[1]:
                        best = (mv, score, cand, cA, cB)
            if best is None:
                print(f"  [{tag}/{Path(path).stem} r{r}] fixed point", flush=True)
                break
            mv, score, cand, cA, cB = best
            bits = cand
            uA, uB = cA, cB
            desc = f"{mv[1]}{mv[2]:+d}" if mv[0] == "single" else f"{mv[1]}+1/{mv[2]}-1"
            print(f"  [{tag}/{Path(path).stem} r{r}] {mv[0]} {desc} gain={score:+.5f} "
                  f"avg={avg(bits):.4f} ({time.time()-t0:.0f}s)", flush=True)
        results[Path(path).stem] = {"bits": bits, "avg_bit": avg(bits),
                                    "uA": float(uA), "uB": float(uB), "u_full": float(uA + uB)}

    if not results:
        print(f"[{tag}] NO STARTS"); return
    best_name = max(results, key=lambda k: results[k]["u_full"])
    bits = results[best_name]["bits"]
    print(f"[{tag}] best: {best_name} u_full={results[best_name]['u_full']:+.5f}", flush=True)

    set_state(bits)
    res, det_t = run_probe_detection(model, loader5k)
    m = eval_probe(res, all_ids, tag="bbox", verbose=False)
    print(f"[{tag}] V3 FINAL 5k: AP={m['AP']:.4f} AP50={m['AP50']:.4f}", flush=True)
    rec = {"best_start": best_name, "bits": bits, "avg_bit": avg(bits),
           "u_full": results[best_name]["u_full"], "AP": m["AP"], "AP50": m["AP50"],
           "all_starts": {k: {"u_full": v["u_full"]} for k, v in results.items()}}
    (OUT / f"{tag}_v3.json").write_text(json.dumps(rec, indent=2))
    applier.restore()
    print(f"EXP55 DONE [{tag}] AP={m['AP']:.4f}", flush=True)


if __name__ == "__main__":
    main()
