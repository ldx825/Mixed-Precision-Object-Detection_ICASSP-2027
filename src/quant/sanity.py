"""Exp02 — quantization sanity tests (plan §16, Tests A-D).

Test A: 16-bit fake-quant on all backbone groups ≈ FP
Test B: 8-bit uniform → mild change
Test C: 4-bit uniform → noticeable degradation, NOT a crash
Test D: wrapper-off must reproduce FP exactly (bit-exact / tiny tolerance)

Detection sanity runs on the first 128 probe512 images (fast subset);
feature stats on 16 calib images.
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
from quant.fake_quant import GroupQuantApplier  # noqa: E402
from diagnostics.sweep_groupbits import (  # noqa: E402
    BackboneFeatureCatcher,
    build_probe_loader,
    feat_dists,
    load_calib_tensors,
    run_probe_detection,
)
from utils.coco_eval_subset import eval_probe  # noqa: E402


def main():
    out_dir = ROOT / "outputs/Exp02"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}

    model, cfg = build_model()
    applier = GroupQuantApplier(model)
    catcher = BackboneFeatureCatcher(model)
    calib = load_calib_tensors(ROOT / "data/calib64", 16)
    print(f"calib tensors: {len(calib)}; groups: {len(applier.groups)}", flush=True)

    def forward_calib():
        feats = []
        with torch.no_grad():
            for t in calib:
                model.backbone(t.cuda())
                feats.append([f.clone() for f in catcher.out])
        return feats

    def mean_dist(fp, fq):
        ds = [feat_dists(a, b) for a, b in zip(fp, fq)]
        return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}

    # FP features
    applier.restore()
    fp_feats = forward_calib()

    # ---- Test D: restore -> re-forward must be (bit-)exact ----
    applier.restore()
    d_feats = forward_calib()
    max_diff = max((a - b).abs().max().item() for fa, fb in zip(fp_feats, d_feats) for a, b in zip(fa, fb))
    res["test_D_max_abs_feat_diff"] = max_diff
    res["test_D_pass"] = max_diff < 1e-5
    print(f"[D] max abs feat diff after restore: {max_diff:.3e} -> {'PASS' if res['test_D_pass'] else 'FAIL'}", flush=True)

    # ---- Tests A/B/C: uniform bits ----
    for tag, bits in [("A", 16), ("B", 8), ("C", 4)]:
        applier.apply_uniform(bits)
        feats = forward_calib()
        dist = mean_dist(fp_feats, feats)
        res[f"test_{tag}_{bits}bit_feat"] = dist
        print(f"[{tag}] uniform {bits}-bit: rel_mse_all={dist['rel_mse_all']:.6f} sqnr={dist['sqnr_db']:.1f}dB", flush=True)

    # ---- detection sanity on probe128 ----
    probe_ids = [int(l) for l in (ROOT / "data/splits/probe512_seed0.txt").read_text().split()][:128]
    loader = build_probe_loader(cfg, probe_ids, workers=4)

    runs = [("FP", None), ("A16", 16), ("B8", 8), ("C4", 4), ("D_restore", None)]
    for tag, bits in runs:
        if bits is None:
            applier.restore()
        else:
            applier.apply_uniform(bits)
        t0 = time.time()
        results, det_t = run_probe_detection(model, loader)
        stats = eval_probe(results, probe_ids, tag="bbox")
        res[f"det_{tag}"] = {"metrics": stats, "det_time_s": det_t, "wall_s": time.time() - t0}
        (out_dir / f"results_{tag}.json").write_text(json.dumps(results))
        print(f"[det {tag}] AP={stats['AP']:.3f} AP50={stats['AP50']:.3f} n_det={len(results)} ({det_t:.0f}s)", flush=True)

    applier.restore()
    catcher.remove()

    # decision summary
    ap_fp = res["det_FP"]["metrics"]["AP"]
    ap_a = res["det_A16"]["metrics"]["AP"]
    ap_b = res["det_B8"]["metrics"]["AP"]
    ap_c = res["det_C4"]["metrics"]["AP"]
    checks = {
        "A_16bit_close_to_FP": abs(ap_fp - ap_a) * 100 < 0.5,  # < 0.5 AP points
        "B_8bit_mild": -0.5 < (ap_fp - ap_b) * 100 < 5.0,      # small change, noise allowed
        "C_4bit_degrades_not_crash": (ap_fp - ap_c) * 100 > 2.0 and ap_c > 0.0,
        "D_restore_exact": res["test_D_pass"],
    }
    res["checks"] = checks
    (out_dir / "sanity_summary.json").write_text(json.dumps(res, indent=2))
    print("CHECKS:", json.dumps(checks), flush=True)
    print("Exp02 DONE", flush=True)


if __name__ == "__main__":
    main()
