"""Exp30b — full C(12,3)=220 triples + C(12,4)=495 quadruples (resumable, shares runs.json).

Same contract as Exp30 (U6 base, repair to 8-bit, calib64, churn vs FP teacher).
Runs are appended to outputs/Exp30_pair_interaction/runs.json so Exp30 results are reused.
Quadruples enable |S|=3 context analysis in Exp31.
"""
import itertools
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


def main():
    out_root = ROOT / "outputs/Exp30_pair_interaction"
    out_root.mkdir(parents=True, exist_ok=True)
    runs_file = out_root / "runs.json"
    runs = json.loads(runs_file.read_text()) if runs_file.exists() else {}

    model, cfg = build_model()
    applier = GroupQuantApplier(model)
    blocks, stg = get_block_and_stage_groups(model)
    pd = dict(model.named_parameters())
    all_names = {**blocks, **stg}
    blocks_sorted = sorted(blocks)

    def set_state(bits_map):
        applier.restore()
        for gname, bits in bits_map.items():
            for n in all_names[gname]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        torch.cuda.synchronize()

    from pycocotools.coco import COCO

    coco = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
    cat_ids = coco.getCatIds()

    applier.restore()
    fp_dets = run_calib_dets(model, cat_ids)
    if "BASE" not in runs:
        set_state({g: 6 for g in blocks})
        runs["BASE"] = churn(fp_dets, run_calib_dets(model, cat_ids))
        runs_file.write_text(json.dumps(runs, indent=2))
        print("[BASE] done", flush=True)

    triples = list(itertools.combinations(blocks_sorted, 3))
    quads = list(itertools.combinations(blocks_sorted, 4))

    def run_combo(combo):
        key = "+".join(combo)
        if key in runs:
            return True
        t0 = time.time()
        bits_map = {**{g: 6 for g in blocks}, **{g: 8 for g in combo}}
        set_state(bits_map)
        ch = churn(fp_dets, run_calib_dets(model, cat_ids))
        runs[key] = ch
        runs_file.write_text(json.dumps(runs, indent=2))
        print(f"[{key}] score_drop={ch['score_drop']:+.4f} ({time.time()-t0:.0f}s)", flush=True)
        return True

    todo = triples + quads
    done0 = len(runs)
    for c in todo:
        run_combo(c)
    applier.restore()
    print(f"EXP30B DONE: {len(runs)-done0} new runs, total keys={len(runs)}")


if __name__ == "__main__":
    main()
