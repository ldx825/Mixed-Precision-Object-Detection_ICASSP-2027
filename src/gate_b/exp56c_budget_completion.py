"""exp56c — deterministic budget completion from existing 6.0 allocations.

Pre-registered rule: after refinement reaches its fixed point, repeatedly apply the
(i:+1, j:-1) candidate with the LARGEST avg-bit increase that fits the remaining
budget (0 < dAvg <= target - avg); ties broken by the full-64 utility (argmax).
This spends leftover budget under a fixed, utility-consistent rule.

Starts (tolerant to missing files):
  - outputs/GateB/eval/cfgf7850b3a/bits.json   (exp40 6.0 snapshot, avg 5.9487, AP 41.95)
  - outputs/GateB/power_refined2/6.0_pair.json (pair-refined, avg 5.9487, AP 41.93)
  - outputs/GateB/cond_full_60.json            (exp56 full greedy output, if present)
Each completion is 5k-evaluated (dedup by identical bits). The deployable output is
the argmax-64-utility variant.

Outputs: outputs/GateB/power_refined2/6.0_complete_<stem>.json
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
TARGET = 6.0
STARTS = [
    "outputs/GateB/eval/cfgf7850b3a/bits.json",
    "outputs/GateB/power_refined2/6.0_pair.json",
    "outputs/GateB/cond_full_60.json",
]


def load_bits(path):
    if not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    if "bits" in d:
        return {g: int(v) for g, v in d["bits"].items()}
    for v in d.values():
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

    results = {}
    seen = {}
    t0 = time.time()
    for path in STARTS:
        bits = load_bits(path)
        if bits is None:
            print(f"[complete] skip missing start {path}", flush=True)
            continue
        stem = Path(path).parent.name if Path(path).name == "bits.json" else Path(path).stem
        print(f"[complete/{stem}] start avg={avg(bits):.4f}", flush=True)
        uA, uB = U_split(bits)
        rounds = 0
        while True:
            rem = TARGET - avg(bits)
            cands = []
            for i, j in permutations(blocks_sorted, 2):
                if bits[i] >= 8 or bits[j] <= 4:
                    continue
                cand = dict(bits)
                cand[i] += 1
                cand[j] -= 1
                davg = avg(cand) - avg(bits)
                if 1e-12 < davg <= rem + 1e-9:
                    cands.append((davg, i, j, cand))
            if not cands:
                break
            maxd = max(c[0] for c in cands)
            ties = [c for c in cands if abs(c[0] - maxd) < 1e-12]
            if len(ties) == 1:
                davg, i, j, cand = ties[0]
                cA, cB = U_split(cand)
            else:
                best_t = None
                for davg_t, i_t, j_t, cand_t in ties:
                    cA_t, cB_t = U_split(cand_t)
                    u_t = cA_t + cB_t
                    if best_t is None or u_t > best_t[0]:
                        best_t = (u_t, (davg_t, i_t, j_t, cand_t), cA_t, cB_t)
                _, (davg, i, j, cand), cA, cB = best_t
            bits = cand
            uA, uB = cA, cB
            rounds += 1
            print(f"  [complete/{stem} r{rounds}] {i}+1/{j}-1 davg=+{davg:.4f} avg={avg(bits):.4f} ({time.time()-t0:.0f}s)", flush=True)
        key = json.dumps({g: bits[g] for g in blocks_sorted}, sort_keys=True)
        if key in seen:
            print(f"[complete/{stem}] duplicate of {seen[key]}; skipping 5k", flush=True)
            continue
        seen[key] = stem
        set_state(bits)
        res, det_t = run_probe_detection(model, loader5k)
        m = eval_probe(res, all_ids, tag="bbox", verbose=False)
        print(f"[complete/{stem}] COMPLETE 5k: AP={m['AP']:.4f} avg={avg(bits):.4f}", flush=True)
        results[stem] = {"bits": bits, "avg_bit": avg(bits), "uA": float(uA), "uB": float(uB),
                         "u_full": float(uA + uB), "AP": m["AP"], "AP50": m["AP50"],
                         "rounds": rounds, "source": path}
        (OUT / f"6.0_complete_{stem}.json").write_text(json.dumps(results[stem], indent=2))

    if results:
        best = max(results, key=lambda k: results[k]["u_full"])
        print(f"[complete] deployable (64-utility argmax): {best} "
              f"u_full={results[best]['u_full']:+.5f} AP={results[best]['AP']:.4f}", flush=True)
    applier.restore()
    print("EXP56C DONE", flush=True)


if __name__ == "__main__":
    main()
