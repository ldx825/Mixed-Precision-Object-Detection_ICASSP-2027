"""exp65 — baseline fairness extended to 4.0 and 5.5 (level set {3..8}).

Same procedure as exp64 (3-bit signals already measured; greedy from all-3
base for F-LBQ-reimpl feature-MSE and BLOBQ-reimpl output-damage curves),
only TARGETS differ. Completes the fairness matrix so every budget has a
strongest-baseline number under the extended contract.

Output: outputs/GateB/power_refined2/{tag}_flbq3.json / {tag}_blobq3.json
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
sys.path.insert(0, str(ROOT / "src"))

from baseline.exp01_repro_fp import build_model  # noqa: E402
from diagnostics.sweep_v2 import get_block_and_stage_groups  # noqa: E402
from quant.fake_quant import GroupQuantApplier, quantize_weight_tensor  # noqa: E402
from diagnostics.sweep_groupbits import build_probe_loader, run_probe_detection  # noqa: E402
from utils.coco_eval_subset import eval_probe  # noqa: E402

OUT = ROOT / "outputs/GateB/power_refined2"
SIGF = ROOT / "outputs/GateB_signals/signals.json"
BITS = [3, 4, 5, 6, 7, 8]
TARGETS = ["4.0", "5.5"]


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
    all_ids = sorted(int(i) for i in coco.getImgIds())
    loader5k = build_probe_loader(cfg, all_ids, workers=6)

    sig = json.loads(SIGF.read_text())
    recs = sig["records"]

    def set_state(bvec):
        applier.restore()
        for g, bits in bvec.items():
            for n in all_names[g]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        torch.cuda.synchronize()

    def avg(b):
        return sum(counts[g] * b[g] for g in blocks_sorted) / total

    def curve(g, key, sub):
        return {b: float(recs[f"{g}@{b}"][key][sub]) for b in BITS}

    def greedy(curves, target):
        b = {g: BITS[0] for g in blocks_sorted}
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
            set_state(b)
            res, det_t = run_probe_detection(model, loader5k)
            m = eval_probe(res, all_ids, tag="bbox", verbose=False)
            print(f"[{tag}-{name}3] AP={m['AP']:.4f} avg={avg(b):.4f} bits={b}", flush=True)
            (OUT / f"{tag}_{name}3.json").write_text(json.dumps(
                {"bits": b, "avg_bit": avg(b), "AP": m["AP"], "AP50": m["AP50"]}, indent=2))
    applier.restore()
    print("EXP65 DONE", flush=True)


if __name__ == "__main__":
    main()
