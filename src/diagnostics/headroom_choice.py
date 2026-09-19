"""One-shot headroom check (plan §35-36 quick version).

Budget: repair exactly K=4 blocks from 4bit->8bit on top of the all-4bit base,
compare selection strategies by ACTUAL probe512 AP:
  FEAT : top-4 by D_feat (F-LBQ-style: larger feature alteration gets more bits)
  OUT  : top-4 by legal output signal (BASE4->repair lost_frac; prior probe
         showed spearman +0.893 with repair benefit)
  ORAC : top-4 by true leave-up benefit (upper-bound reference)
  RAND : seeded random 4 blocks

Reference points: BASE4 AP 17.52, CHK_all8 ~42.7, FP 43.02 (probe512).
"""
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
sys.path.insert(0, str(ROOT / "src"))

from baseline.exp01_repro_fp import build_model  # noqa: E402
from diagnostics.sweep_v2 import get_block_and_stage_groups  # noqa: E402
from quant.fake_quant import GroupQuantApplier, quantize_weight_tensor  # noqa: E402
from diagnostics.sweep_groupbits import build_probe_loader, run_probe_detection  # noqa: E402
from diagnostics.actionability_probe import load_dets, churn  # noqa: E402
from utils.coco_eval_subset import eval_probe  # noqa: E402

K = 4


def main():
    out_root = ROOT / "outputs/Exp03v2_headroom"
    out_root.mkdir(exist_ok=True)

    # ---- strategy inputs from existing data ----
    summary = json.loads((ROOT / "tables/Exp03v2_all_summary.json").read_text())
    ap = {"FP": summary["AP_FP_probe512"], "BASE4": summary["AP_base4"]}
    D, benefit = {}, {}
    for r in summary["designs"]["leave-up"]["rows"]:
        D[r["group"]] = r["D"]
        benefit[r["group"]] = r["AP"] - ap["BASE4"]

    base_dets = load_dets(ROOT / "outputs/Exp03v2/BASE4/results_bbox.json")
    up_churn = {}
    for g in D:
        up_dets = load_dets(ROOT / f"outputs/Exp03v2/UP_{g}/results_bbox.json")
        up_churn[g] = churn(base_dets, up_dets)  # BASE4 -> repair output change

    sel_feat = sorted(D, key=lambda g: -D[g])[:K]
    sel_out = sorted(D, key=lambda g: -up_churn[g]["lost_frac"])[:K]
    sel_orac = sorted(D, key=lambda g: -benefit[g])[:K]
    rng = random.Random(0)
    sel_rand = sorted(rng.sample(list(D), K))

    strategies = {"FEAT": sel_feat, "OUT": sel_out, "ORAC": sel_orac, "RAND": sel_rand}
    print("strategies:")
    for k_, v in strategies.items():
        print(f"  {k_:5s}: {v}")

    # ---- model ----
    model, cfg = build_model()
    applier = GroupQuantApplier(model)
    blocks, stg = get_block_and_stage_groups(model)
    pd = dict(model.named_parameters())
    all_names = {**blocks, **stg}

    def set_state(bits_map):
        applier.restore()
        for gname, bits in bits_map.items():
            for n in all_names[gname]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        torch.cuda.synchronize()

    probe_ids = [int(l) for l in (ROOT / "data/splits/probe512_seed0.txt").read_text().split()][:512]
    loader = build_probe_loader(cfg, probe_ids, workers=6)

    results_eval = {}
    for name, sel in strategies.items():
        tag = f"{name}_K{K}"
        d = out_root / tag
        if not (d / "done.json").exists():
            d.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            set_state({**{g: 4 for g in blocks}, **{g: 8 for g in sel}})
            res, det_t = run_probe_detection(model, loader)
            (d / "results_bbox.json").write_text(json.dumps(res))
            (d / "done.json").write_text(json.dumps({"sel": sel, "det_s": det_t}))
            print(f"[{tag}] inferred {len(res)} dets in {det_t:.0f}s", flush=True)
        else:
            print(f"[skip-infer] {tag}")
        res = json.loads((d / "results_bbox.json").read_text())
        m = eval_probe(res, sorted({r["image_id"] for r in res}), tag="bbox", verbose=False)
        results_eval[tag] = m
        print(f"[{tag}] AP={m['AP']:.4f} AP50={m['AP50']:.4f}", flush=True)

    applier.restore()

    print("\n=== HEADROOM SUMMARY (probe512 bbox AP) ===")
    print(f"  FP      = {ap['FP']:.4f}  (reference)")
    print(f"  BASE4   = {ap['BASE4']:.4f}  (all 4bit)")
    for name in strategies:
        print(f"  {name:6s}  = {results_eval[f'{name}_K{K}']['AP']:.4f}   sel={strategies[name]}")

    (out_root / "headroom_metrics.json").write_text(json.dumps({
        "strategies": strategies, "metrics": results_eval,
        "ref": ap, "up_churn": up_churn,
    }, indent=2))
    print("\nwritten outputs/Exp03v2_headroom/headroom_metrics.json")
    print("HEADROOM DONE")


if __name__ == "__main__":
    main()
