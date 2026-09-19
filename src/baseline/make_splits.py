"""Create calib64 (train2017 subset via HF streaming) and probe512 (val2017) splits.

calib64: 64 images randomly sampled from COCO train2017 (seed 0), matching the
F-LBQ supplementary recipe ("randomly select 64 samples from training set").
Implemented via HuggingFace streaming of `detection-datasets/coco` (no full
train2017 download feasible on this disk — DECISIONS.md D003).

probe512: 512 val2017 images (seed 0) for stage-1 sweeps (plan §20).
"""
import json
import random
from pathlib import Path

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")


def make_probe512():
    from pycocotools.coco import COCO

    ann = str(ROOT / "data/coco/annotations/instances_val2017.json")
    coco = COCO(ann)
    ids = sorted(coco.getImgIds())
    rng = random.Random(0)
    probe = sorted(rng.sample(ids, 512))
    out = ROOT / "data/splits/probe512_seed0.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(i) for i in probe))
    print(f"probe512: {len(probe)} ids -> {out}")
    return probe


def make_calib64():
    """Sample 64 train2017 images (seed 0) from local HF parquet shards."""
    import io
    import random

    import pyarrow.parquet as pq
    from PIL import Image

    parquet_dir = ROOT / "data/hf_parquet"
    shards = sorted(parquet_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards in {parquet_dir}")
    rows = []
    for f in shards:
        t = pq.read_table(f, columns=["image", "image_id"])
        for r in t.to_pylist():
            rows.append(r)
    print(f"total rows across {len(shards)} shards: {len(rows)}")

    rng = random.Random(0)
    sel = sorted(rng.sample(range(len(rows)), 64))

    out_dir = ROOT / "data/calib64"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for i, idx in enumerate(sel):
        row = rows[idx]
        img_field = row["image"]
        b = img_field["bytes"] if isinstance(img_field, dict) else img_field
        img = Image.open(io.BytesIO(b)).convert("RGB")
        iid = int(row.get("image_id", -1))
        img.save(out_dir / f"calib_{i:03d}_id{iid}.png")
        ids.append(iid)
    split = ROOT / "data/splits/calib64_seed0.txt"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text("\n".join(str(i) for i in ids))
    print(f"calib64: {len(ids)} images -> {out_dir}, ids -> {split}")


if __name__ == "__main__":
    import sys

    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("probe", "both"):
        make_probe512()
    if what in ("calib", "both"):
        make_calib64()
