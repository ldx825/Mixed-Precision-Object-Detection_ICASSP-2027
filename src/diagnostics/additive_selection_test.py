"""Additive-prior selection test: what an independent-sensitivity method would actually pick.

Additive/independent family (BLOB-Q-style Gamma_i, LampQ-style Fisher, F-LBQ-style feature
alteration) ranks blocks by a SINGLETON score and takes top-K. The singleton scores we have:

  s1 (block-only damage, FP base, {g}->4bit) : top4 = [BLK1.0, BLK2.5, BLK3.0, BLK3.1]
  s2 (D_feat, F-LBQ score)                   : top4 = [BLK0.0, BLK1.0, BLK0.1, BLK1.1]  (= FEAT, already run)

This script runs the s1-selected allocation (repair BLK1.0/2.5/3.0/3.1 to 8bit on all-4bit base)
on probe512 and 5k, to compare against conditional selections (OUT512 34.42, OUTCALsd 34.95,
ORAC 35.32, FEAT 22.76, BASE4 17.67).
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
from diagnostics.sweep_groupbits import build_probe_loader, run_probe_detection  # noqa: E402
from utils.coco_eval_subset import eval_probe  # noqa: E402

SEL_S1 = ["BLK1.0", "BLK2.5", "BLK3.0", "BLK3.1"]  # top4 by singleton damage (FP base)


def main():
    out_root = ROOT / "outputs/Exp04_additive_test"
    out_root.mkdir(exist_ok=True)

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

    from pycocotools.coco import COCO

    coco = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))

    bits_map = {**{g: 4 for g in blocks}, **{g: 8 for g in SEL_S1}}
    results = {}
    for split, tag in [("probe", "probe512"), ("full", "5k")]:
        d = out_root / tag
        if not (d / "done.json").exists():
            ids = ([int(l) for l in (ROOT / "data/splits/probe512_seed0.txt").read_text().split()][:512]
                   if split == "probe" else sorted(int(i) for i in coco.getImgIds()))
            loader = build_probe_loader(cfg, ids, workers=6)
            set_state(bits_map)
            res, det_t = run_probe_detection(model, loader)
            d.mkdir(parents=True, exist_ok=True)
            (d / "results_bbox.json").write_text(json.dumps(res))
            (d / "done.json").write_text(json.dumps({"sel": SEL_S1, "det_s": det_t}))
        res = json.loads((d / "results_bbox.json").read_text())
        m = eval_probe(res, sorted({r["image_id"] for r in res}), tag="bbox", verbose=False)
        results[tag] = m
        print(f"[{tag}] SEL_S1 AP={m['AP']:.4f} AP50={m['AP50']:.4f}", flush=True)

    applier.restore()
    (out_root / "metrics.json").write_text(json.dumps(results, indent=2))
    print("\n=== comparison (same budget, repair 4/12 blocks) ===")
    print(f"  BASE4          : 17.67 / 17.52 (5k/probe512)")
    print(f"  FEAT (D_feat)  : 22.76 / 22.55")
    print(f"  SEL_S1 (indep) : {results['5k']['AP']*100:.2f} / {results['probe512']['AP']*100:.2f}")
    print(f"  RAND (probe512): 31.07 (5k n/a)")
    print(f"  OUT512 (cond)  : 34.42 / 34.45")
    print(f"  OUTCALsd(cond) : 34.95 / n/a")
    print(f"  ORAC (cond)    : 35.32 / 35.80")
    print("ADDITIVE_TEST DONE")


if __name__ == "__main__":
    main()
