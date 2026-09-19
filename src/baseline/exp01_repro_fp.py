"""Exp01 — FP32 Swin-T Mask R-CNN reproduction on COCO val2017.

Plan §15: reproduce official model-zoo metrics (box AP ~46.0, mask AP ~41.6).
Gate: deviation > 0.5 AP => investigate; > 1.0 AP => STOP (do not proceed).

Runs the official mmdet 3.3.0 config `mask-rcnn_swin-t-p4-w7_fpn_1x_coco`
with the official 1x checkpoint on all 5000 val2017 images.
Saves COCO-format results (bbox + segm) and metrics json.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")


def build_model(cfg_path=None, ckpt=None):
    import mmdet
    from mmengine.config import Config
    from mmdet.apis import init_detector

    mmdet_dir = Path(os.path.dirname(mmdet.__file__))
    if cfg_path is None:
        candidates = [
            mmdet_dir / "configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py",
            ROOT / "third_party/mmdetection/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py",
        ]
        cfg_path = next((c for c in candidates if Path(c).exists()), None)
        if cfg_path is None:
            raise FileNotFoundError("swin mask-rcnn config not found (check third_party/mmdetection clone)")
    if ckpt is None:
        ckpt = ROOT / "checkpoints/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth"
    cfg = Config.fromfile(str(cfg_path))
    # strip init_cfg (backbone pretrained URLs) — full checkpoint is loaded right after
    def _strip(node):
        if isinstance(node, dict):
            if node.get("init_cfg", None) is not None:
                node["init_cfg"] = None
            for v in node.values():
                _strip(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _strip(v)
    _strip(cfg)
    # ensure custom imports resolve (config references mmdet classes)
    if cfg.get("default_scope") is None:
        cfg.default_scope = "mmdet"
    model = init_detector(cfg, str(ckpt), device="cuda:0")
    model.eval()
    return model, cfg


def encode_rle(mask_np):
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask_np.astype(np.uint8)))
    return {"size": rle["size"], "counts": rle["counts"].decode("ascii")}


def sample_to_coco(sample, img_id, cat_ids=None):
    """Convert a DetDataSample to COCO result dicts (bbox + segm).

    cat_ids: official COCO category ids ordered as mmdet labels (dataset.cat_ids).
    """
    inst = sample.pred_instances
    bboxes = inst.bboxes.detach().cpu().numpy()
    scores = inst.scores.detach().cpu().numpy()
    labels = inst.labels.detach().cpu().numpy()
    masks = None
    if getattr(inst, "masks", None) is not None:
        masks = inst.masks.detach().cpu().numpy()

    bbox_res, segm_res = [], []
    for i in range(len(scores)):
        x1, y1, x2, y2 = bboxes[i].tolist()
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        cid = int(cat_ids[int(labels[i])]) if cat_ids is not None else int(labels[i]) + 1
        det = {
            "image_id": int(img_id),
            "category_id": cid,
            "score": float(scores[i]),
            "bbox": [x1, y1, w, h],
        }
        bbox_res.append(det)
        if masks is not None:
            segm = dict(det)
            segm["segmentation"] = encode_rle(masks[i])
            segm_res.append(segm)
    return bbox_res, segm_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/Exp01"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all 5000")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=500, help="checkpoint partial results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from mmengine.config import Config
    import mmdet
    from mmdet.registry import DATASETS

    t_start = time.time()
    model, cfg = build_model()

    # ---- dataset from official config test pipeline ----
    mmdet_dir = Path(os.path.dirname(mmdet.__file__))
    ds_cfg = cfg.test_dataloader.dataset.copy()
    ds_cfg["data_root"] = str(ROOT / "data/coco")
    ds_cfg["ann_file"] = "annotations/instances_val2017.json"
    ds_cfg["data_prefix"] = dict(img="val2017/")
    ds_cfg["test_mode"] = True
    ds_cfg["metainfo"] = cfg.get("metainfo", ds_cfg.get("metainfo", None))
    dataset = DATASETS.build(ds_cfg)
    if args.limit:
        dataset = torch.utils.data.Subset(dataset, list(range(args.limit)))
    base_ds = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    cat_ids = base_ds.cat_ids  # official COCO ids ordered by mmdet label index

    from mmengine.dataset import pseudo_collate

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=pseudo_collate,
        pin_memory=False,
    )

    bbox_all, segm_all = [], []
    timing = []
    n_done = 0
    t0 = time.time()
    with torch.no_grad():
        for i, data in enumerate(loader):
            tick = time.time()
            sample = data["data_samples"][0]
            img_id = int(sample.metainfo["img_id"])
            result = model.test_step(data)
            b_res, s_res = sample_to_coco(result[0], img_id, cat_ids)
            bbox_all.extend(b_res)
            segm_all.extend(s_res)
            timing.append(time.time() - tick)
            n_done += 1
            if i % 100 == 0:
                speed = n_done / (time.time() - t0 + 1e-9)
                eta = (len(loader) - n_done) / max(speed, 1e-6)
                print(f"[{i}/{len(loader)}] {speed:.2f} img/s ETA {eta/60:.1f} min", flush=True)
            if args.save_every and n_done % args.save_every == 0:
                (out_dir / "partial_bbox.json").write_text(json.dumps(bbox_all))
                (out_dir / "partial_segm.json").write_text(json.dumps(segm_all))

    (out_dir / "results_bbox.json").write_text(json.dumps(bbox_all))
    (out_dir / "results_segm.json").write_text(json.dumps(segm_all))
    (out_dir / "timing.json").write_text(
        json.dumps({"n": n_done, "total_s": time.time() - t0, "per_img_s": timing[:50],
                    "wall_clock_start": t_start})
    )

    # ---- COCO eval ----
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ann = str(ROOT / "data/coco/annotations/instances_val2017.json")
    coco_gt = COCO(ann)

    metrics = {}
    for tag, res in [("bbox", bbox_all), ("segm", segm_all)]:
        if not res:
            continue
        coco_dt = coco_gt.loadRes(res)
        ev = COCOeval(coco_gt, coco_dt, tag)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        m = ev.stats
        metrics[tag] = {
            "AP": m[0], "AP50": m[1], "AP75": m[2],
            "AP_S": m[3], "AP_M": m[4], "AP_L": m[5],
        }
        print(tag, json.dumps(metrics[tag], indent=2))

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("DONE", json.dumps(metrics))


if __name__ == "__main__":
    main()
