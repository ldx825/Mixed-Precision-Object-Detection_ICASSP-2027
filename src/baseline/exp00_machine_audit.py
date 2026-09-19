"""Exp00 — machine & asset audit. Saves everything needed for reproducibility."""
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")


def sha256_head(path, nbytes=10 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def main():
    out = ROOT / "outputs/Exp00"
    out.mkdir(parents=True, exist_ok=True)
    info = {"date": "2026-09-12", "host": platform.node(), "python": sys.version}

    import torch
    info["torch"] = torch.__version__
    info["cuda_runtime"] = torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version()
    info["gpu"] = torch.cuda.get_device_name(0)
    info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)

    for mod in ["torchvision", "mmcv", "mmdet", "mmengine", "numpy", "pycocotools"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception as e:
            info[mod] = f"ERR: {e}"

    # checkpoints
    ckpt = ROOT / "checkpoints/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth"
    info["ckpt"] = {
        "path": str(ckpt),
        "exists": ckpt.exists(),
        "size": ckpt.stat().st_size if ckpt.exists() else None,
        "sha256_head10MB": sha256_head(ckpt) if ckpt.exists() else None,
    }

    # data
    coco = ROOT / "data/coco"
    info["data"] = {
        "val2017_imgs": len(list((coco / "val2017").glob("*.jpg"))) if (coco / "val2017").exists() else 0,
        "ann_instances_val": (coco / "annotations/instances_val2017.json").exists(),
        "calib64_imgs": len(list((ROOT / "data/calib64").glob("*.png"))),
        "probe512_split": (ROOT / "data/splits/probe512_seed0.txt").exists(),
    }
    s = (ROOT / "data/splits/probe512_seed0.txt")
    if s.exists():
        info["data"]["probe512_n"] = len(s.read_text().split())

    info["disk_free_gb"] = round(os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize / 1e9, 1)

    (out / "machine_audit.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print("Exp00 DONE")


if __name__ == "__main__":
    main()
