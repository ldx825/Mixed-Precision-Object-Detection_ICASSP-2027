"""Gate B — singleton signal measurement (FP context).

Configs (all on calib64, FP reference):
  FP                                   (reference)
  60x single-block (g,b): only block g fake-quantized to b, rest FP
  36x perturbation (g,t): only block g perturbed (N(0, (0.02*std_w)^2)), rest FP
Per config: churn vs FP (score_drop / lost_frac / new_frac / iou_drop)
          + backbone feature rel_mse vs FP.
Output: outputs/GateB_signals/signals.json (includes per-block param counts)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
sys.path.insert(0, str(ROOT / "src"))

from baseline.exp01_repro_fp import build_model  # noqa: E402
from diagnostics.sweep_v2 import get_block_and_stage_groups  # noqa: E402
from quant.fake_quant import GroupQuantApplier, quantize_weight_tensor  # noqa: E402
from diagnostics.sweep_groupbits import BackboneFeatureCatcher, feat_dists, load_calib_tensors  # noqa: E402
from diagnostics.actionability_probe import churn  # noqa: E402
from diagnostics.confirm_5k import run_calib_dets  # noqa: E402

OUT = ROOT / "outputs/GateB_signals"
OUT.mkdir(parents=True, exist_ok=True)

BITS = [4, 5, 6, 7, 8]
PERT_TRIALS = 3
PERT_SIGMA = 0.02


def main():
    model, cfg = build_model()
    applier = GroupQuantApplier(model)
    blocks, stg = get_block_and_stage_groups(model)
    pd = dict(model.named_parameters())
    all_names = {**blocks, **stg}
    blocks_sorted = sorted(blocks)
    counts = {g: sum(pd[n].numel() for n in blocks[g]) for g in blocks_sorted}

    from pycocotools.coco import COCO

    coco = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
    cat_ids = coco.getCatIds()

    calib = load_calib_tensors(ROOT / "data/calib64", 64)
    catcher = BackboneFeatureCatcher(model)

    def feat_forward():
        feats = []
        with torch.no_grad():
            for t in calib:
                model.backbone(t.cuda())
                feats.append([f.clone() for f in catcher.out])
        return feats

    def set_state(bits_map, perturb=None):
        applier.restore()
        for gname, bits in bits_map.items():
            for n in all_names[gname]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        if perturb is not None:
            g, seed = perturb
            gen = torch.Generator(device="cpu").manual_seed(seed)
            for n in all_names[g]:
                w = pd[n].data
                noise = torch.randn(w.shape, generator=gen) * (PERT_SIGMA * float(w.std().item()))
                w.add_(noise.to(w.device))
        torch.cuda.synchronize()

    records = {}
    t_start = time.time()

    # ---- FP reference ----
    applier.restore()
    torch.cuda.synchronize()
    fp_dets = run_calib_dets(model, cat_ids)
    fp_feats = feat_forward()
    records["FP"] = {"churn": churn(fp_dets, fp_dets), "feat": {"rel_mse_all": 0.0}}
    print(f"[FP] n_fp={records['FP']['churn']['n_fp']} ({time.time()-t_start:.0f}s)", flush=True)

    # ---- single-block curves ----
    for g in blocks_sorted:
        for b in BITS:
            set_state({g: b})
            dets = run_calib_dets(model, cat_ids)
            ch = churn(fp_dets, dets)
            feats = feat_forward()
            ds = [feat_dists(a, b_) for a, b_ in zip(fp_feats, feats)]
            fd = {k: float(np.mean([x[k] for x in ds])) for k in ds[0]}
            records[f"{g}@{b}"] = {"churn": ch, "feat": fd}
            print(f"[{g}@{b}] score_drop={ch['score_drop']:+.5f} lost={ch['lost_frac']:.4f} "
                  f"rel_mse={fd['rel_mse_all']:.5f} ({time.time()-t_start:.0f}s)", flush=True)

    # ---- perturbation sensitivity (LampQ-style Fisher proxy) ----
    for g in blocks_sorted:
        for t in range(PERT_TRIALS):
            set_state({}, perturb=(g, 1000 + 7 * t))
            dets = run_calib_dets(model, cat_ids)
            ch = churn(fp_dets, dets)
            records[f"{g}~{t}"] = {"churn": ch}
            print(f"[{g}~{t}] score_drop={ch['score_drop']:+.5f} ({time.time()-t_start:.0f}s)", flush=True)

    applier.restore()
    payload = {"counts": counts, "bits": BITS, "pert_sigma": PERT_SIGMA,
               "pert_trials": PERT_TRIALS, "records": records}
    (OUT / "signals.json").write_text(json.dumps(payload, indent=2))
    print(f"SIGNALS DONE ({time.time()-t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
