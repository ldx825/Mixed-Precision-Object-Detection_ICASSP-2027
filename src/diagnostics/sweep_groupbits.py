"""Exp03 sweep driver — single-group weight quantization perturbation.

For each (group, bits) in Swin-T block linear groups x {6,4}:
  1. quantize only that group's weights (fake-quant), everything else FP
  2. measure backbone feature distortion D_feat over calib64 (plan §21)
  3. run detection on probe512 (bbox results saved as COCO json)

Outputs (per run):  outputs/Exp03/<group>__b<bits>/
    feat_dist.json   — relative MSE per stage + pooled, cosine
    results_bbox.json — COCO bbox detections on probe512
    run_meta.json    — timing, bits, group, done flag

Baseline FP run uses the same pipeline with no quantization (group="FP").

Usage:
  python sweep_groupbits.py --bits 6,4 --probe 512 --resume
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
sys.path.insert(0, str(ROOT / "src"))

from quant.fake_quant import GroupQuantApplier  # noqa: E402

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# ---------------- preprocessing (mmdet-equivalent: resize(1333,800) -> pad/32 -> normalize) ----------------

def preprocess_uint8(img_u8: np.ndarray) -> np.ndarray:
    """Replicate mmdet test pipeline Resize(keep_ratio)+Pad(32)+Normalize -> CHW float32."""
    h, w = img_u8.shape[:2]
    max_long, max_short = 1333, 800
    if w >= h:
        scale = min(max_long / w, max_short / h)
    else:
        scale = min(max_long / h, max_short / w)
    nh, nw = int(h * scale), int(w * scale)
    pil = Image.fromarray(img_u8).resize((nw, nh), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.uint8)
    # pad to /32
    pad_h = (32 - nh % 32) % 32
    pad_w = (32 - nw % 32) % 32
    arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
    x = arr.astype(np.float32)
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]  # 1CHW


def load_calib_tensors(calib_dir: Path, limit: int = 64):
    files = sorted(calib_dir.glob("*.png"))[:limit]
    return [torch.from_numpy(preprocess_uint8(np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8))) for f in files]


# ---------------- feature hooks ----------------

class BackboneFeatureCatcher:
    def __init__(self, model):
        self.out = None
        self.h = model.backbone.register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        self.out = [t.detach().clone() for t in (o if isinstance(o, (tuple, list)) else [o])]

    def remove(self):
        self.h.remove()


def feat_dists(f_fp, f_q):
    out = {}
    for si, (a, b) in enumerate(zip(f_fp, f_q)):
        num = (a - b).pow(2).sum().item()
        den = a.pow(2).sum().item() + 1e-12
        out[f"rel_mse_stage{si}"] = num / den
    A = torch.cat([t.flatten() for t in f_fp])
    B = torch.cat([t.flatten() for t in f_q])
    out["rel_mse_all"] = ((A - B).pow(2).sum() / (A.pow(2).sum() + 1e-12)).item()
    out["cosine_all"] = torch.nn.functional.cosine_similarity(A[None], B[None], dim=1).item()
    # SQNR: 10*log10(signal/noise)
    sig = A.pow(2).sum().item()
    noi = (A - B).pow(2).sum().item() + 1e-12
    out["sqnr_db"] = 10 * np.log10(sig / noi)
    return out


# ---------------- detection on probe ----------------

def build_probe_loader(cfg, probe_ids, workers=4):
    import os
    import mmdet
    from mmdet.registry import DATASETS
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate

    ds_cfg = cfg.test_dataloader.dataset.copy()
    ds_cfg["data_root"] = str(ROOT / "data/coco")
    ds_cfg["ann_file"] = "annotations/instances_val2017.json"
    ds_cfg["data_prefix"] = dict(img="val2017/")
    ds_cfg["test_mode"] = True
    ds_cfg["metainfo"] = cfg.get("metainfo", ds_cfg.get("metainfo", None))
    dataset = DATASETS.build(ds_cfg)
    id2idx = {int(dataset.get_data_info(i)["img_id"]): i for i in range(len(dataset))}
    idxs = [id2idx[i] for i in probe_ids if i in id2idx]
    subset = torch.utils.data.Subset(dataset, idxs)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=1, shuffle=False, num_workers=workers,
        collate_fn=pseudo_collate, pin_memory=False,
    )
    return loader


def _get_cat_ids(loader):
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return getattr(ds, "cat_ids", None)


def run_probe_detection(model, loader):
    cat_ids = _get_cat_ids(loader)

    results = []
    t0 = time.time()
    n = 0
    with torch.no_grad():
        for data in loader:
            sample = data["data_samples"][0]
            img_id = int(sample.metainfo["img_id"])
            r = model.test_step(data)[0]
            inst = r.pred_instances
            bboxes = inst.bboxes.detach().cpu().numpy()
            scores = inst.scores.detach().cpu().numpy()
            labels = inst.labels.detach().cpu().numpy()
            for i in range(len(scores)):
                x1, y1, x2, y2 = bboxes[i].tolist()
                cid = int(cat_ids[int(labels[i])]) if cat_ids is not None else int(labels[i]) + 1
                results.append({
                    "image_id": img_id,
                    "category_id": cid,
                    "score": float(scores[i]),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                })
            n += 1
            if n % 128 == 0:
                print(f"    probe {n} done ({time.time()-t0:.0f}s)", flush=True)
    return results, time.time() - t0


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", default="6,4")
    ap.add_argument("--groups", default="all", help='all | comma list of group names | "FP"')
    ap.add_argument("--probe", type=int, default=512)
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "outputs/Exp03"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    bits_list = [int(b) for b in args.bits.split(",") if b]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- model ----
    sys.path.insert(0, str(ROOT / "src"))
    from baseline.exp01_repro_fp import build_model  # reuse builder
    model, cfg = build_model()
    applier = GroupQuantApplier(model)
    all_groups = list(applier.groups.keys())
    print(f"groups: {len(all_groups)}", flush=True)

    if args.groups == "all":
        want_groups = all_groups
    elif args.groups == "FP":
        want_groups = []
    else:
        want_groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    # ---- probe loader ----
    probe_ids = [int(l) for l in (ROOT / "data/splits/probe512_seed0.txt").read_text().split()]
    probe_ids = probe_ids[: args.probe]
    loader = build_probe_loader(cfg, probe_ids, workers=args.workers)

    # ---- calib tensors (fp features cached once) ----
    calib_dir = ROOT / "data/calib64"
    calib_tensors = load_calib_tensors(calib_dir, args.calib)
    print(f"calib tensors: {len(calib_tensors)}", flush=True)

    catcher = BackboneFeatureCatcher(model)

    def compute_fp_calib_feats():
        applier.restore()
        feats = []
        with torch.no_grad():
            for t in calib_tensors:
                model.backbone(t.cuda())
                feats.append([f.clone() for f in catcher.out])
        return feats

    print("computing FP calib features ...", flush=True)
    fp_feats = compute_fp_calib_feats()

    def calc_feat_dist():
        fq = []
        with torch.no_grad():
            for t in calib_tensors:
                model.backbone(t.cuda())
                fq.append([f.clone() for f in catcher.out])
        # mean per-image dists
        dists = [feat_dists(a, b) for a, b in zip(fp_feats, fq)]
        keys = dists[0].keys()
        return {k: float(np.mean([d[k] for d in dists])) for k in keys}

    # ---- runs ----
    runs = [("FP", 32)] + [(g, b) for g in want_groups for b in bits_list]
    for gname, bits in runs:
        tag = "FP" if gname == "FP" else f"{gname}__b{bits}"
        run_dir = out_root / tag
        done_flag = run_dir / "done.json"
        if args.resume and done_flag.exists():
            print(f"[skip] {tag}", flush=True)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.time()

        if gname == "FP":
            applier.restore()
        else:
            applier.apply(gname, bits)

        fd = calc_feat_dist()
        (run_dir / "feat_dist.json").write_text(json.dumps(fd, indent=2))
        print(f"[{tag}] feat rel_mse_all={fd['rel_mse_all']:.6f} sqnr={fd['sqnr_db']:.1f}dB", flush=True)

        results, det_time = run_probe_detection(model, loader)
        (run_dir / "results_bbox.json").write_text(json.dumps(results))
        meta = {"group": gname, "bits": bits, "n_det": len(results),
                "det_time_s": det_time, "total_s": time.time() - t_start}
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
        (run_dir / "done.json").write_text(json.dumps({"ok": True}))
        print(f"[{tag}] det_time={det_time:.0f}s n_det={len(results)}", flush=True)

    applier.restore()
    catcher.remove()
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
