"""COCO evaluation restricted to a subset of image ids (probe sets)."""
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
ANN = str(ROOT / "data/coco/annotations/instances_val2017.json")


def eval_probe(results, img_ids, ann_file=None, tag="bbox", verbose=False):
    """COCOeval on a subset of images. Returns metric dict."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ann_file = ann_file or ANN
    coco_gt = COCO(ann_file)
    if not results:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP_S": 0.0, "AP_M": 0.0, "AP_L": 0.0,
                "n_results": 0}
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, tag)
    ev.params.maxDets = [1, 10, 100]
    ev.params.imgIds = sorted(int(i) for i in img_ids)
    ev.evaluate()
    ev.accumulate()
    # NOTE: pycocotools 2.0.11 sets ev.stats only inside summarize()
    import contextlib
    import io

    if verbose:
        ev.summarize()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            ev.summarize()
    m = ev.stats
    return {"AP": float(m[0]), "AP50": float(m[1]), "AP75": float(m[2]),
            "AP_S": float(m[3]), "AP_M": float(m[4]), "AP_L": float(m[5]),
            "n_results": len(results)}


def eval_dir(json_path, img_ids, tag="bbox", ann_file=None):
    results = json.loads(Path(json_path).read_text())
    return eval_probe(results, img_ids, ann_file=ann_file, tag=tag)
