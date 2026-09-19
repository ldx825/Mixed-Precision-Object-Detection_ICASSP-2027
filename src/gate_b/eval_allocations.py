"""Gate B — 5k evaluation of all allocation policies.

Sources: outputs/GateB/allocations_static.json, cond_alloc.json, allocations_oracle.json (if exists)
Each entry: {budget: {method: {bits: {g: b}}}}  -> evaluate unique bit-configs on COCO val2017 5k
(bbox AP; resumable; dedup identical configs across methods/budgets).

Priority order: budget "5.0" first (pilot), then "6.0", "4.5", then others.
Output: outputs/GateB/eval/<config_id>/{results_bbox.json, done.json, metrics.json}
        outputs/GateB/eval_summary.json  ({budget: {method: {config_id, AP, ...}}})
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
sys.path.insert(0, str(ROOT / "src"))

from baseline.exp01_repro_fp import build_model  # noqa: E402
from diagnostics.sweep_v2 import get_block_and_stage_groups  # noqa: E402
from quant.fake_quant import GroupQuantApplier, quantize_weight_tensor  # noqa: E402
from diagnostics.sweep_groupbits import build_probe_loader, run_probe_detection  # noqa: E402
from utils.coco_eval_subset import eval_probe  # noqa: E402

OUT = ROOT / "outputs/GateB"
EVAL = OUT / "eval"
EVAL.mkdir(parents=True, exist_ok=True)
PRIORITY = ["5.0", "4.5", "6.0", "5.5", "4.0"]
METHOD_PRIO = ["conditional-OUT", "static-OUT", "BLOBQ-reimpl-greedy", "oracle-approx",
               "uniform", "BLOBQ-reimpl-DP", "F-LBQ-reimpl", "LampQ-style"]


def load_sources():
    sources = {}
    for fn in ["allocations_static.json", "cond_alloc.json", "allocations_oracle.json"]:
        p = OUT / fn
        if p.exists():
            data = json.loads(p.read_text())
            if fn == "cond_alloc.json":
                # cond_alloc.json is {budget: {bits, avg_bit}} (single method) -> wrap
                data = {b: {"conditional-OUT": v} for b, v in data.items() if "bits" in v}
            sources[fn] = data
    return sources


def collect_configs(sources):
    """-> {(budget, method): config_id}, {config_id: bits}, ordered by priority."""
    entries = []
    for src, data in sources.items():
        for budget, methods in data.items():
            for method, v in methods.items():
                entries.append((budget, method, v["bits"]))
    # order: priority budgets first, then method priority
    def rank(e):
        b, m = e[0], e[1]
        return (PRIORITY.index(b) if b in PRIORITY else len(PRIORITY),
                METHOD_PRIO.index(m) if m in METHOD_PRIO else len(METHOD_PRIO))
    entries.sort(key=rank)
    config_key = lambda bits: json.dumps({g: int(bits[g]) for g in sorted(bits)}, sort_keys=True)
    ck_to_cid = {}
    bits_of = {}
    mapping = []
    for budget, method, bits in entries:
        ck = config_key(bits)
        if ck not in ck_to_cid:
            cid = "cfg" + hashlib.md5(ck.encode()).hexdigest()[:8]
            ck_to_cid[ck] = cid
            bits_of[cid] = {g: int(bits[g]) for g in sorted(bits)}
        mapping.append((budget, method, ck_to_cid[ck]))
    return mapping, ck_to_cid, bits_of


def main():
    sources = load_sources()
    if not sources:
        print("no allocation files found"); return
    mapping, cid_of, bits_of = collect_configs(sources)
    print(f"unique configs to evaluate: {len(bits_of)}")

    model, cfg = build_model()
    applier = GroupQuantApplier(model)
    blocks, stg = get_block_and_stage_groups(model)
    pd = dict(model.named_parameters())
    all_names = {**blocks, **stg}

    def set_state(bits_map):
        applier.restore()
        for gname, bits in bits_map.items():
            for n in all_names[gname]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        torch.cuda.synchronize()

    from pycocotools.coco import COCO

    coco = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
    all_ids = sorted(int(i) for i in coco.getImgIds())
    loader = build_probe_loader(cfg, all_ids, workers=6)

    # evaluate unique configs in order of first appearance (priority-sorted)
    seen = set()
    metrics_all = {}
    for budget, method, cid in mapping:
        if cid in seen:
            continue
        seen.add(cid)
        d = EVAL / cid
        d.mkdir(exist_ok=True)
        if not (d / "done.json").exists():
            bits = bits_of[cid]
            t0 = time.time()
            set_state(bits)
            res, det_t = run_probe_detection(model, loader)
            (d / "results_bbox.json").write_text(json.dumps(res))
            (d / "bits.json").write_text(json.dumps(bits, indent=2))
            (d / "done.json").write_text(json.dumps({"n": len(res), "det_s": det_t}))
            print(f"[{cid}] 5k done {len(res)} dets {det_t:.0f}s", flush=True)
        res = json.loads((d / "results_bbox.json").read_text())
        m = eval_probe(res, all_ids, tag="bbox", verbose=False)
        (d / "metrics.json").write_text(json.dumps(m, indent=2))
        metrics_all[cid] = m
        print(f"[{cid}] AP={m['AP']:.4f} AP50={m['AP50']:.4f} AP75={m['AP75']:.4f} "
              f"AP_S={m['AP_S']:.3f} AP_M={m['AP_M']:.3f} AP_L={m['AP_L']:.3f}", flush=True)
        # refresh summary
        summary = {}
        for b2, m2, cid2 in mapping:
            if cid2 in metrics_all:
                summary.setdefault(b2, {})[m2] = {"config_id": cid2, **metrics_all[cid2]}
        (OUT / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    applier.restore()
    print("EVAL DONE")


if __name__ == "__main__":
    main()
