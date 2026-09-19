"""exp64 — fairness check: singleton baselines with the level set extended to {3..8}.

Step 1: measure 3-bit signals (churn + backbone feature rel-mse) for the 12
        Swin-T blocks; the existing signals.json covers {4..8} only.
Step 2: re-run F-LBQ-reimpl (feature-MSE gain/cost greedy) and
        BLOBQ-reimpl-greedy (output-damage gain/cost greedy) allocators with
        BITS = {3,...,8}, upgrading from the all-3 base to each target budget.
Step 3: 5k-evaluate the resulting allocations at 4.5 and 5.0.

Purpose: verify whether simply granting the level-3 permission to singleton-
based allocators recovers the margin (it should not, because their singleton
scoring cannot see the context-dependent value of the 3-bit move).

Output: outputs/GateB/power_refined2/{tag}_flbq3.json / {tag}_blobq3.json
"""
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
from diagnostics.sweep_groupbits import BackboneFeatureCatcher, feat_dists, load_calib_tensors  # noqa: E402
from diagnostics.actionability_probe import churn  # noqa: E402
from diagnostics.confirm_5k import run_calib_dets  # noqa: E402
from diagnostics.sweep_groupbits import build_probe_loader, run_probe_detection  # noqa: E402
from utils.coco_eval_subset import eval_probe  # noqa: E402

OUT = ROOT / "outputs/GateB/power_refined2"
OUT.mkdir(parents=True, exist_ok=True)
SIGF = ROOT / "outputs/GateB_signals/signals.json"
BITS = [3, 4, 5, 6, 7, 8]
TARGETS = ["4.5", "5.0"]


def main():
    model, cfg = build_model()
    applier = GroupQuantApplier(model)
    blocks, stg = get_block_and_stage_groups(model)
    pd = dict(model.named_parameters())
    all_names = {**blocks, **stg}
    blocks_sorted = sorted(blocks)
    counts = {g: sum(pd[n].numel() for n in blocks[g]) for g in blocks_sorted}
    total = sum(counts.values())

    from pycocotools.coco import COCO

    coco = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
    cat_ids = coco.getCatIds()
    all_ids = sorted(int(i) for i in coco.getImgIds())
    loader5k = build_probe_loader(cfg, all_ids, workers=6)

    calib = load_calib_tensors(ROOT / "data/calib64", 64)
    catcher = BackboneFeatureCatcher(model)

    def set_state(bvec):
        applier.restore()
        for g, bits in bvec.items():
            for n in all_names[g]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        torch.cuda.synchronize()

    def feat_forward():
        feats = []
        with torch.no_grad():
            for t in calib:
                model.backbone(t.cuda())
                feats.append([f.clone() for f in catcher.out])
        return feats

    def avg(b):
        return sum(counts[g] * b[g] for g in blocks_sorted) / total

    def eval5k(b, tag, out_name):
        set_state(b)
        res, det_t = run_probe_detection(model, loader5k)
        m = eval_probe(res, all_ids, tag="bbox", verbose=False)
        print(f"[{tag}] 5k: AP={m['AP']:.4f} AP50={m['AP50']:.4f} avg={avg(b):.4f}", flush=True)
        (OUT / out_name).write_text(json.dumps(
            {"bits": b, "avg_bit": avg(b), "AP": m["AP"], "AP50": m["AP50"]}, indent=2))
        return m

    # ---- Step 1: measure 3-bit signals, merge into signals.json ----
    sig = json.loads(SIGF.read_text())
    recs = sig["records"]
    missing3 = [g for g in blocks_sorted if f"{g}@3" not in recs]
    if missing3:
        print(f"[signals] measuring 3-bit for {len(missing3)} blocks...", flush=True)
        applier.restore()
        fp_dets = run_calib_dets(model, cat_ids)
        fp_feats = feat_forward()
        t0 = time.time()
        for g in missing3:
            applier.restore()
            for n in all_names[g]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, 3))
            torch.cuda.synchronize()
            ch = churn(fp_dets, run_calib_dets(model, cat_ids))
            feats = feat_forward()
            ds = [feat_dists(a, b_) for a, b_ in zip(fp_feats, feats)]
            fd = {"rel_mse_all": float(sum(d["rel_mse_all"] for d in ds) / max(1, len(ds)))}
            recs[f"{g}@3"] = {"churn": ch, "feat": fd}
            print(f"  [3bit {g}] score_drop={ch['score_drop']:.5f} rel_mse={fd['rel_mse_all']:.5f} ({time.time()-t0:.0f}s)", flush=True)
        sig["bits"] = BITS
        SIGF.write_text(json.dumps(sig, indent=2))
        print("[signals] updated", flush=True)
    else:
        print("[signals] 3-bit already present", flush=True)

    # ---- Step 2: greedy allocation with {3..8} for F-LBQ (feat) and BLOBQ (churn) ----
    def curve(g, key, sub=None):
        out = {}
        for b in BITS:
            r = recs[f"{g}@{b}"]
            out[b] = float(r["feat"][sub]) if key == "feat" else float(r["churn"][sub])
        return out

    def greedy(curves, target):
        b = {g: BITS[0] for g in blocks_sorted}   # start at all-3
        while True:
            rem = target - avg(b)
            best = None
            for g in blocks_sorted:
                if b[g] >= BITS[-1]:
                    continue
                if counts[g] / total > rem + 1e-9:
                    continue
                gain = curves[g][b[g]] - curves[g][b[g] + 1]
                if gain <= 0:
                    continue
                ratio = gain / counts[g]
                if best is None or ratio > best[1]:
                    best = (g, ratio)
            if best is None:
                break
            b[best[0]] += 1
        return b

    for tag in TARGETS:
        T = float(tag)
        for name, key, sub in [("flbq", "feat", "rel_mse_all"), ("blobq", "churn", "score_drop")]:
            curves = {g: curve(g, key, sub) for g in blocks_sorted}
            b = greedy(curves, T)
            print(f"[{tag}-{name}3] alloc avg={avg(b):.4f} bits={b}", flush=True)
            eval5k(b, f"{tag}-{name}3", f"{tag}_{name}3.json")

    applier.restore()
    print("EXP64 DONE", flush=True)


if __name__ == "__main__":
    main()
