"""Exp03 v2 — block-level perturbation designs (v1 in-linear unit was below threshold).

Designs:
  leave-up : base = ALL blocks quantized to 4bit. For each block g: raise g to 8bit.
             D_feat(g) = ||F_up(g) - F_base||^2 / ||F_up(g)||^2   (mean over calib64)
             Benefit(g) = AP_base - AP_up(g)   (AP recovery when block g is repaired)
  stage    : quantize all blocks of ONE stage to {6,4}, rest FP (independent disturbance)
  block-only: quantize ONE block to {6,4}, rest FP (independent disturbance)

Run tags: BASE4 | UP_<blk> | STG<i>__b<b> | BLK<i>.<j>__b<b>
Outputs: outputs/Exp03v2/<tag>/{feat_dist.json, results_bbox.json, run_meta.json, done.json}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
sys.path.insert(0, str(ROOT / "src"))

from quant.fake_quant import GroupQuantApplier, quantize_weight_tensor  # noqa: E402
from diagnostics.sweep_groupbits import (  # noqa: E402
    BackboneFeatureCatcher,
    build_probe_loader,
    feat_dists,
    load_calib_tensors,
    run_probe_detection,
)


def get_block_and_stage_groups(model):
    """dicts: block name -> [param names]; stage name -> [param names]."""
    pd = dict(model.named_parameters())
    backbone = model.backbone
    stages_attr = "stages" if hasattr(backbone, "stages") else "layers"
    stages = getattr(backbone, stages_attr)
    blocks, stg = {}, {}
    for li, stage in enumerate(stages):
        for bi in range(len(stage.blocks)):
            prefix = f"backbone.{stages_attr}.{li}.blocks.{bi}"
            names = [
                f"{prefix}.attn.w_msa.qkv.weight",
                f"{prefix}.attn.w_msa.proj.weight",
                f"{prefix}.ffn.layers.0.0.weight",
                f"{prefix}.ffn.layers.1.weight",
            ]
            names = [n for n in names if n in pd]
            bname = f"BLK{li}.{bi}"
            blocks[bname] = names
            stg.setdefault(f"STG{li}", []).extend(names)
    return blocks, stg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="leave-up", choices=["leave-up", "stage", "block-only"])
    ap.add_argument("--bits", default="6,4")  # for stage/block-only
    ap.add_argument("--base-bits", type=int, default=4)
    ap.add_argument("--probe", type=int, default=512)
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "outputs/Exp03v2"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    bits_list = [int(b) for b in args.bits.split(",") if b]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    from baseline.exp01_repro_fp import build_model

    model, cfg = build_model()
    applier = GroupQuantApplier(model)
    blocks, stg = get_block_and_stage_groups(model)
    pd = dict(model.named_parameters())
    all_names = {**blocks, **stg}
    print(f"blocks={len(blocks)} stages={len(stg)}", flush=True)

    def set_state(bits_map):
        applier.restore()
        for gname, bits in bits_map.items():
            for n in all_names[gname]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        torch.cuda.synchronize()

    probe_ids = [int(l) for l in (ROOT / "data/splits/probe512_seed0.txt").read_text().split()][: args.probe]
    loader = build_probe_loader(cfg, probe_ids, workers=args.workers)
    calib_tensors = load_calib_tensors(ROOT / "data/calib64", args.calib)
    catcher = BackboneFeatureCatcher(model)

    def forward_calib():
        feats = []
        with torch.no_grad():
            for t in calib_tensors:
                model.backbone(t.cuda())
                feats.append([f.clone() for f in catcher.out])
        return feats

    def mean_feat_dist(a_feats, b_feats):
        ds = [feat_dists(a, b) for a, b in zip(a_feats, b_feats)]
        return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}

    def do_run(tag, bits_map, ref_feats, meta_extra):
        run_dir = out_root / tag
        if args.resume and (run_dir / "done.json").exists():
            print(f"[skip] {tag}", flush=True)
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        set_state(bits_map)
        cur = forward_calib()
        fd = mean_feat_dist(cur, ref_feats) if ref_feats is not None else mean_feat_dist(cur, cur)
        fd["_self"] = 1.0 if ref_feats is None else 0.0
        (run_dir / "feat_dist.json").write_text(json.dumps(fd, indent=2))
        print(f"[{tag}] rel_mse_all={fd['rel_mse_all']:.6f} sqnr={fd['sqnr_db']:.1f}dB", flush=True)
        results, det_t = run_probe_detection(model, loader)
        (run_dir / "results_bbox.json").write_text(json.dumps(results))
        meta = {"group": meta_extra.pop("group", None), "bits": meta_extra.pop("bits", None),
                "design": args.design, "n_det": len(results), "det_time_s": det_t,
                "total_s": time.time() - t0, **meta_extra}
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
        (run_dir / "done.json").write_text(json.dumps({"ok": True}))
        print(f"[{tag}] det={det_t:.0f}s n_det={len(results)}", flush=True)

    if args.design == "leave-up":
        base_map = {g: args.base_bits for g in blocks}
        base_dir = out_root / "BASE4"
        have_base = (base_dir / "done.json").exists()
        if not (args.resume and have_base):
            do_run("BASE4", base_map, None, {"group": "BASE", "bits": args.base_bits,
                                             "note": "uniform 4bit all blocks"})
        # (re)compute base feats in-memory
        set_state(base_map)
        base_feats = forward_calib()
        # fp feats for diagnostics
        applier.restore()
        fp_feats = forward_calib()
        fd_base_vs_fp = mean_feat_dist(fp_feats, base_feats)
        print(f"[BASE4] vs FP: rel_mse_all={fd_base_vs_fp['rel_mse_all']:.6f} sqnr={fd_base_vs_fp['sqnr_db']:.1f}dB", flush=True)
        (out_root / "BASE4_vs_fp.json").write_text(json.dumps(fd_base_vs_fp, indent=2))
        for g in blocks:
            bits_map = {**base_map, g: 8}
            do_run(f"UP_{g}", bits_map, base_feats,
                   {"group": g, "bits": "4->8(up)", "base_bits": args.base_bits})
    elif args.design == "stage":
        applier.restore()
        fp_feats = forward_calib()
        for sname in stg:
            for b in bits_list:
                do_run(f"{sname}__b{b}", {sname: b}, fp_feats,
                       {"group": sname, "bits": b})
    elif args.design == "block-only":
        applier.restore()
        fp_feats = forward_calib()
        for bname in blocks:
            for b in bits_list:
                do_run(f"{bname}__b{b}", {bname: b}, fp_feats,
                       {"group": bname, "bits": b})

    applier.restore()
    catcher.remove()
    print("SWEEP_V2 DONE", flush=True)


if __name__ == "__main__":
    main()
