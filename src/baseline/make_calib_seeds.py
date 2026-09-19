"""Generate 5 calibration seed pools (256 images each, nested prefixes) for Addendum §19.

Each pool: rng=Random(seed); perm=shuffle(range(N_rows)); take first 256 indices.
n = 8,16,32,64,128,256 -> the first-n images of the pool (nested prefixes).
Images stored as raw JPEG bytes (same source as calib64_seed0: HF coco train2017 parquet).
"""
import random
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")


def main():
    parquet_dir = ROOT / "data/hf_parquet"
    shards = sorted(parquet_dir.glob("train-*.parquet"))
    rows = []
    for f in shards:
        t = pq.read_table(f, columns=["image", "image_id"])
        rows.extend(t.to_pylist())
    print(f"total rows: {len(rows)}", flush=True)

    for seed in range(5):
        rng = random.Random(seed)
        perm = list(range(len(rows)))
        rng.shuffle(perm)
        sel = perm[:256]
        out = ROOT / f"data/calib_seeds/seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        ids = []
        for i, idx in enumerate(sel):
            row = rows[idx]
            b = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
            (out / f"calib_{i:03d}.jpg").write_bytes(b)
            ids.append(int(row.get("image_id", -1)))
        (ROOT / f"data/splits/calib256_seed{seed}.txt").write_text("\n".join(map(str, ids)))
        print(f"seed{seed}: 256 imgs -> {out}", flush=True)

    print("CALIB_SEEDS DONE")


if __name__ == "__main__":
    main()
