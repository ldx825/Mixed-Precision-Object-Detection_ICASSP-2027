"""Evaluate all Exp03 sweep runs on their probe ids; produce summary table.

Outputs:
  tables/Exp03_summary.json  — per-run: feat dist + probe metrics + deltas vs FP
  tables/Exp03_summary.csv   — same, flat
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
sys.path.insert(0, str(ROOT / "src"))

from utils.coco_eval_subset import eval_probe  # noqa: E402


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=str(ROOT / "outputs/Exp03"))
    ap.add_argument("--out-prefix", default="Exp03")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    tables = ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([d for d in sweep_dir.iterdir() if d.is_dir() and (d / "results_bbox.json").exists()])
    if not run_dirs:
        print("no runs found")
        return

    # FP baseline
    fp_dir = sweep_dir / "FP"
    fp_metrics = None
    if fp_dir.exists():
        results = json.loads((fp_dir / "results_bbox.json").read_text())
        img_ids = sorted({r["image_id"] for r in results})
        fp_metrics = eval_probe(results, img_ids, tag="bbox")
        print("FP:", json.dumps(fp_metrics), flush=True)

    rows = []
    for d in run_dirs:
        meta = json.loads((d / "run_meta.json").read_text()) if (d / "run_meta.json").exists() else {}
        feat = json.loads((d / "feat_dist.json").read_text()) if (d / "feat_dist.json").exists() else {}
        results = json.loads((d / "results_bbox.json").read_text())
        img_ids = sorted({r["image_id"] for r in results})
        m = eval_probe(results, img_ids, tag="bbox")
        row = {
            "tag": d.name, "group": meta.get("group"), "bits": meta.get("bits"),
            **{f"feat_{k}": v for k, v in feat.items()},
            **{f"AP_{k}": v for k, v in m.items()},
        }
        if fp_metrics:
            row["dAP"] = fp_metrics["AP"] - m["AP"]
            for k in ["AP50", "AP75", "AP_S", "AP_M", "AP_L"]:
                row[f"d{k}"] = fp_metrics[k] - m[k]
        rows.append(row)
        print(f"[{d.name}] AP={m['AP']:.3f} dAP={row.get('dAP', float('nan')):+.3f} "
              f"rel_mse={feat.get('rel_mse_all', float('nan')):.6f}", flush=True)

    (tables / f"{args.out_prefix}_summary.json").write_text(json.dumps({"fp": fp_metrics, "rows": rows}, indent=2))
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(tables / f"{args.out_prefix}_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"summary written: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
