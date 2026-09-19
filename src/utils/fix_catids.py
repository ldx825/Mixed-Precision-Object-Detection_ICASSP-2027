"""Fix category ids in saved Exp01 results (labels+1 -> official COCO ids) and re-evaluate.

The original run stored category_id = mmdet_label + 1, which is wrong for COCO
(official ids have gaps: 12, 26, 29, 30, 45, 66, 68-72 are unused).
Mapping: official_id = coco.getCatIds()[label].
"""
import json
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")


def main():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ann = str(ROOT / "data/coco/annotations/instances_val2017.json")
    coco = COCO(ann)
    cat_ids = coco.getCatIds()
    print(f"official cat ids: {len(cat_ids)}")

    out = ROOT / "outputs/Exp01"
    metrics = {}
    for tag in ["bbox", "segm"]:
        p = out / f"results_{tag}.json"
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        for r in res:
            r["category_id"] = int(cat_ids[r["category_id"] - 1])
        (out / f"results_{tag}_fixed.json").write_text(json.dumps(res))
        dt = coco.loadRes(res)
        ev = COCOeval(coco, dt, tag)
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        m = ev.stats
        metrics[tag] = {"AP": float(m[0]), "AP50": float(m[1]), "AP75": float(m[2]),
                        "AP_S": float(m[3]), "AP_M": float(m[4]), "AP_L": float(m[5])}
        print(tag, json.dumps(metrics[tag], indent=2), flush=True)

    (out / "metrics_fixed.json").write_text(json.dumps(metrics, indent=2))
    ref = {"box": 46.0, "mask": 41.6}
    for tag, key in [("bbox", "box"), ("segm", "mask")]:
        if tag in metrics:
            d = metrics[tag]["AP"] * 100 - ref[key]
            print(f"deviation {key}: {d:+.2f} AP -> {'OK' if abs(d) <= 0.5 else ('CHECK' if abs(d) <= 1.0 else 'STOP')}")
    print("FIXED DONE")


if __name__ == "__main__":
    main()
