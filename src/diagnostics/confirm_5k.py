"""Post-5k confirmations (runs automatically after eval5k finishes).

Part A — deployable selection check: compute BASE4->repair output churn using
         calib64 (64 unlabeled train images only), select top-4; compare with
         probe512-based selection and oracle; report churn-rank agreement.
Part B — 5k evaluation of FEAT / OUT(probe512) / OUT(calib64) / ORAC combos.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
sys.path.insert(0, str(ROOT / "src"))

from baseline.exp01_repro_fp import build_model  # noqa: E402
from diagnostics.sweep_v2 import get_block_and_stage_groups  # noqa: E402
from quant.fake_quant import GroupQuantApplier, quantize_weight_tensor  # noqa: E402
from diagnostics.sweep_groupbits import build_probe_loader, run_probe_detection  # noqa: E402
from diagnostics.actionability_probe import churn  # noqa: E402
from utils.coco_eval_subset import eval_probe  # noqa: E402

K = 4


def run_calib_dets(model, cat_ids):
    from mmdet.apis import inference_detector

    pngs = sorted((ROOT / "data/calib64").glob("*.png"))
    by = {}
    with torch.no_grad():
        for i, p in enumerate(pngs):
            img = np.asarray(Image.open(p).convert("RGB"))
            r = inference_detector(model, img)
            inst = r.pred_instances
            b = inst.bboxes.cpu().numpy()
            s = inst.scores.cpu().numpy()
            lb = inst.labels.cpu().numpy()
            for j in range(len(s)):
                x1, y1, x2, y2 = b[j].tolist()
                by.setdefault(i, []).append({
                    "image_id": i, "category_id": int(cat_ids[int(lb[j])]),
                    "score": float(s[j]),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                })
    return by


def main():
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
    cat_ids = coco.getCatIds()

    # ---------------- Part A: calib64-based selection ----------------
    print("=== Part A: calib64-based deployable selection ===", flush=True)
    set_state({g: 4 for g in blocks})
    base_c = run_calib_dets(model, cat_ids)
    calib_churn = {}
    for g in blocks:
        set_state({**{gg: 4 for gg in blocks}, g: 8})
        up_c = run_calib_dets(model, cat_ids)
        calib_churn[g] = churn(base_c, up_c)
        print(f"[calib {g}] lost={calib_churn[g]['lost_frac']:.3f} "
              f"score_drop={calib_churn[g]['score_drop']:+.3f}", flush=True)

    sel_calib = sorted(blocks, key=lambda g: -calib_churn[g]["lost_frac"])[:K]
    # deployable signal v2: largest score increase upon repair (calib64, also no-GT)
    sel_calib_sd = sorted(blocks, key=lambda g: calib_churn[g]["score_drop"])[:K]
    print("sel_calib_sd :", sorted(sel_calib_sd), flush=True)

    hr = json.loads((ROOT / "outputs/Exp03v2_headroom/headroom_metrics.json").read_text())
    up_churn_probe = hr["up_churn"]
    from scipy import stats
    xs = [calib_churn[g]["lost_frac"] for g in blocks]
    ys = [up_churn_probe[g]["lost_frac"] for g in blocks]
    sp = stats.spearmanr(xs, ys)
    print(f"\ncalib64-vs-probe512 churn rank agreement: spearman={sp.statistic:+.3f} (p={sp.pvalue:.3f})")
    print("sel_calib   :", sorted(sel_calib))
    print("sel_probe512:", sorted(hr["strategies"]["OUT"]))
    print("sel_oracle  :", sorted(hr["strategies"]["ORAC"]))
    print("sel_feat    :", sorted(hr["strategies"]["FEAT"]))

    # ---------------- Part B: 5k combos ----------------
    print("\n=== Part B: 5k evaluation of combos ===", flush=True)
    # heal cfg test pipeline if contaminated by inference_detector (ndarray branch mutates cfg!)
    pl = cfg.test_dataloader.dataset.pipeline
    if str(pl[0].get("type", "")).endswith("LoadImageFromNDArray"):
        pl[0]["type"] = "LoadImageFromFile"
        print("(cfg test_pipeline[0] healed -> LoadImageFromFile)", flush=True)
    all_ids = sorted(int(i) for i in coco.getImgIds())
    loader5k = build_probe_loader(cfg, all_ids, workers=6)
    combos = {
        "FEAT": hr["strategies"]["FEAT"],
        "OUT512": hr["strategies"]["OUT"],
        "OUTCALsd": sorted(sel_calib_sd),
        "ORAC": hr["strategies"]["ORAC"],
    }
    out_root = ROOT / "outputs/Exp03v2_headroom_5k"
    out_root.mkdir(exist_ok=True)
    res_eval = {}
    for name, sel in combos.items():
        d = out_root / name
        if not (d / "done.json").exists():
            d.mkdir(parents=True, exist_ok=True)
            set_state({**{g: 4 for g in blocks}, **{g: 8 for g in sel}})
            res, det_t = run_probe_detection(model, loader5k)
            (d / "results_bbox.json").write_text(json.dumps(res))
            (d / "done.json").write_text(json.dumps({"sel": sel, "det_s": det_t}))
        res = json.loads((d / "results_bbox.json").read_text())
        m = eval_probe(res, sorted({r["image_id"] for r in res}), tag="bbox", verbose=False)
        res_eval[name] = m
        print(f"[5k {name:7s}] AP={m['AP']:.4f} AP50={m['AP50']:.4f} sel={sel}", flush=True)

    applier.restore()
    (out_root / "metrics_5k.json").write_text(json.dumps(res_eval, indent=2))
    (out_root / "selection_meta.json").write_text(json.dumps({
        "sel_calib": sel_calib, "sel_calib_sd": sel_calib_sd, "calib_churn": calib_churn,
        "churn_agreement_spearman": float(sp.statistic),
    }, indent=2))
    print("\nCONFIRM5K DONE")


if __name__ == "__main__":
    main()
