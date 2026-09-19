"""Exp04 — realistic-budget (uniform-6 base) allocation-direction test.

Question: at a realistic average bit budget (6.0, parameter-weighted), does the
DIRECTION of allocation between equal-size blocks matter, and can deployable
(no-GT) signals pick the better direction?

Stage 1: from uniform-6 base (all 12 blocks at 6bit):
  - per block g: DOWN (g->4), UP (g->8)  [24 runs]
  - per config: D_feat (calib64, vs FP), calib-churn vs FP teacher (legal),
    probe512 AP
  - => realistic-budget versions of the Actionability correlations

Stage 2: equal-size swaps at constant parameter-weighted budget:
  pairs: (BLK3.0, BLK3.1) [7.08M each], (BLK2.3, BLK2.4) [1.77M each]
  direction X = {a->8, b->4}; direction Y = {a->4, b->8}  (both keep avg=6)
  Predictions: ORAC (single-block deltas, additive), OUT (calib churn of DOWN),
               FEAT (D_feat of DOWN). Then run BOTH directions and compare.
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
from diagnostics.sweep_groupbits import (  # noqa: E402
    BackboneFeatureCatcher,
    build_probe_loader,
    feat_dists,
    load_calib_tensors,
    run_probe_detection,
)
from diagnostics.actionability_probe import churn  # noqa: E402
from diagnostics.confirm_5k import run_calib_dets  # noqa: E402
from utils.coco_eval_subset import eval_probe  # noqa: E402

PAIRS = [("BLK3.0", "BLK3.1"), ("BLK2.3", "BLK2.4")]


def main():
    from pycocotools.coco import COCO

    out_root = ROOT / "outputs/Exp04_budget6"
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

    probe_ids = [int(l) for l in (ROOT / "data/splits/probe512_seed0.txt").read_text().split()][:512]
    loader = build_probe_loader(cfg, probe_ids, workers=6)
    coco = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
    cat_ids = coco.getCatIds()

    calib = load_calib_tensors(ROOT / "data/calib64", 64)
    catcher = BackboneFeatureCatcher(model)

    def forward_calib_feats():
        feats = []
        with torch.no_grad():
            for t in calib:
                model.backbone(t.cuda())
                feats.append([f.clone() for f in catcher.out])
        return feats

    # FP references
    applier.restore()
    fp_feats = forward_calib_feats()
    fp_calib_dets = run_calib_dets(model, cat_ids)

    def do_run(tag, bits_map):
        d = out_root / tag
        if not (d / "done.json").exists():
            d.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            set_state(bits_map)
            # feature dist vs FP
            cur = forward_calib_feats()
            ds = [feat_dists(a, b) for a, b in zip(fp_feats, cur)]
            fd = {k: float(np.mean([x[k] for x in ds])) for k in ds[0]}
            # legal churn vs FP teacher on calib64
            cd = run_calib_dets(model, cat_ids)
            ch = churn(fp_calib_dets, cd)
            # probe512 detection
            res, det_t = run_probe_detection(model, loader)
            (d / "feat_dist.json").write_text(json.dumps(fd, indent=2))
            (d / "calib_churn.json").write_text(json.dumps(ch, indent=2))
            (d / "results_bbox.json").write_text(json.dumps(res))
            (d / "done.json").write_text(json.dumps({"det_s": det_t, "total_s": time.time() - t0}))
            print(f"[{tag}] rel_mse={fd['rel_mse_all']:.5f} lost={ch['lost_frac']:.3f} "
                  f"score_drop={ch['score_drop']:+.3f} det={det_t:.0f}s", flush=True)
        m = eval_probe(json.loads((d / "results_bbox.json").read_text()),
                       sorted({r["image_id"] for r in json.loads((d / "results_bbox.json").read_text())}),
                       tag="bbox", verbose=False)
        return m

    def load_side(tag):
        d = out_root / tag
        return {"feat": json.loads((d / "feat_dist.json").read_text()),
                "churn": json.loads((d / "calib_churn.json").read_text())}

    # ---------- Stage 1 ----------
    print("=== Stage 1: single-block perturbation on U6 base ===", flush=True)
    m_u6 = do_run("U6", {g: 6 for g in blocks})
    print(f"[U6] AP={m_u6['AP']:.4f}", flush=True)
    m = {}
    for g in blocks:
        m[("DOWN", g)] = do_run(f"DOWN_{g}", {**{gg: 6 for gg in blocks}, g: 4})
        m[("UP", g)] = do_run(f"UP_{g}", {**{gg: 6 for gg in blocks}, g: 8})
        print(f"[{g}] DOWN AP={m[('DOWN',g)]['AP']:.4f} UP AP={m[('UP',g)]['AP']:.4f}", flush=True)

    # realistic-budget actionability table
    from scipy import stats
    print("\n=== Actionability at 6-bit base (48 single-block configs) ===", flush=True)
    tags = [(k, g) for k in ("DOWN", "UP") for g in blocks]
    dmg = [m_u6["AP"] - m[k]["AP"] for k in tags]
    for name, f in [
        ("D_feat", lambda k, s: s["feat"]["rel_mse_all"]),
        ("lost_frac", lambda k, s: s["churn"]["lost_frac"]),
        ("score_drop", lambda k, s: s["churn"]["score_drop"]),
        ("iou_drop", lambda k, s: s["churn"]["iou_drop"]),
    ]:
        xs = []
        for k in tags:
            side = load_side(("DOWN_" if k[0] == "DOWN" else "UP_") + k[1])
            xs.append(f(k, side))
        sp = stats.spearmanr(xs, dmg)
        print(f"  {name:11s} spearman={sp.statistic:+.3f} p={sp.pvalue:.4f}", flush=True)

    # ---------- Stage 2: equal-size swaps ----------
    print("\n=== Stage 2: equal-size swaps at constant budget ===", flush=True)
    swap_out = {}
    for a, b in PAIRS:
        d_down_a = m_u6["AP"] - m[("DOWN", a)]["AP"]
        d_down_b = m_u6["AP"] - m[("DOWN", b)]["AP"]
        d_up_a = m[("UP", a)]["AP"] - m_u6["AP"]
        d_up_b = m[("UP", b)]["AP"] - m_u6["AP"]
        # X = {a->8, b->4}
        pred_X = d_up_a - d_down_b  # benefit of X vs U6 (additive approx)
        pred_Y = d_up_b - d_down_a
        orac = "X" if pred_X > pred_Y else "Y"
        s_down_a, s_down_b = load_side(f"DOWN_{a}"), load_side(f"DOWN_{b}")
        out = "X" if s_down_a["churn"]["lost_frac"] > s_down_b["churn"]["lost_frac"] else "Y"
        feat = "X" if s_down_a["feat"]["rel_mse_all"] > s_down_b["feat"]["rel_mse_all"] else "Y"
        # run both directions
        mX = do_run(f"SWAP_{a}hi_{b}lo", {**{gg: 6 for gg in blocks}, a: 8, b: 4})
        mY = do_run(f"SWAP_{a}lo_{b}hi", {**{gg: 6 for gg in blocks}, a: 4, b: 8})
        true_dir = "X" if mX["AP"] > mY["AP"] else "Y"
        gap = abs(mX["AP"] - mY["AP"]) * 100
        swap_out[f"{a}|{b}"] = {
            "X": {"sel": f"{a}:8,{b}:4", "AP": mX["AP"]}, "Y": {"sel": f"{a}:4,{b}:8", "AP": mY["AP"]},
            "prediction": {"ORAC": orac, "OUT": out, "FEAT": feat}, "true": true_dir, "gap_AP_points": gap,
        }
        print(f"[{a} vs {b}] X AP={mX['AP']:.4f} vs Y AP={mY['AP']:.4f} -> true={true_dir} gap={gap:.2f} AP pts; "
              f"ORAC={orac} OUT={out} FEAT={feat}", flush=True)

    applier.restore()
    catcher.remove()

    summary = {"U6_AP": m_u6["AP"],
               "single_block": {f"{k[0]}_{k[1]}": m[k]["AP"] for k in m},
               "swaps": swap_out}
    (out_root / "budget6_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nwritten outputs/Exp04_budget6/budget6_summary.json")
    print("BUDGET6 DONE")


if __name__ == "__main__":
    main()
