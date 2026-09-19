"""exp63 — Swin-S deep push: FP baseline + level-3 pair refine from cond45.

  1. FP 5k baseline for the full Swin-S picture.
  2. Level-3 pair-reallocation refinement (i:+1/j:-1, bits[j] > 3, avg <= 4.5,
     epsilon-tolerant split-half acceptance) from the cond45 output (44.66),
     up to 3 rounds -> 5k.
  3. If cond60 (exp61) exists, a completion-style pass at 6.0 (net-increase
     pair moves) -> 5k when it changes anything.

Output: outputs/GateB_swins/{fp_5k.json, cond45_refined.json, ...}
"""
import json
import sys
import time
from itertools import permutations
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
sys.path.insert(0, str(ROOT / "src"))

from baseline.exp01_repro_fp import build_model  # noqa: E402
from diagnostics.sweep_v2 import get_block_and_stage_groups  # noqa: E402
from quant.fake_quant import GroupQuantApplier, quantize_weight_tensor  # noqa: E402
from diagnostics.confirm_5k import run_calib_dets  # noqa: E402
from diagnostics.actionability_probe import churn  # noqa: E402
from diagnostics.sweep_groupbits import build_probe_loader, run_probe_detection  # noqa: E402
from utils.coco_eval_subset import eval_probe  # noqa: E402

OUT = ROOT / "outputs/GateB_swins"
OUT.mkdir(parents=True, exist_ok=True)
BIT_LO, BIT_HI = 3, 8
EPS = 1e-3

CKPT = ROOT / "checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
CFG_CANDS = [
    ROOT / "third_party/mmdetection/configs/swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py",
]


def main():
    cfg_path = next((c for c in CFG_CANDS if Path(c).exists()), None)
    model, cfg = build_model(cfg_path=str(cfg_path), ckpt=str(CKPT))
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

    def set_state(bvec):
        applier.restore()
        for g, bits in bvec.items():
            for n in all_names[g]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        torch.cuda.synchronize()

    def avg(b):
        return sum(counts[g] * b[g] for g in blocks_sorted) / total

    def eval5k(b, tag, out_name=None):
        set_state(b)
        res, det_t = run_probe_detection(model, loader5k)
        m = eval_probe(res, all_ids, tag="bbox", verbose=False)
        print(f"[{tag}] 5k: AP={m['AP']:.4f} AP50={m['AP50']:.4f} avg={avg(b):.4f}", flush=True)
        if out_name:
            (OUT / out_name).write_text(json.dumps(
                {"bits": b, "avg_bit": avg(b), "AP": m["AP"], "AP50": m["AP50"]}, indent=2))
        return m

    # ---- 1) FP baseline (no quantization) ----
    applier.restore()
    res_t, det_t = run_probe_detection(model, loader5k)
    m_fp = eval_probe(res_t, all_ids, tag="bbox", verbose=False)
    print(f"[swins-fp] 5k: AP={m_fp['AP']:.4f} AP50={m_fp['AP50']:.4f}", flush=True)
    (OUT / "fp_5k.json").write_text(json.dumps({"AP": m_fp["AP"], "AP50": m_fp["AP50"]}, indent=2))

    fp64 = run_calib_dets(model, cat_ids)
    fpA = {k: v for k, v in fp64.items() if k < 32}
    fpB = {k: v for k, v in fp64.items() if k >= 32}

    def U_split(b):
        set_state(b)
        dets = run_calib_dets(model, cat_ids)
        A = {k: v for k, v in dets.items() if k < 32}
        B = {k: v for k, v in dets.items() if k >= 32}
        return -churn(fpA, A)["score_drop"], -churn(fpB, B)["score_drop"]

    # ---- 2) level-3 refine from cond45 ----
    p = OUT / "cond45_5k.json"
    if p.exists():
        bits = {g: int(v) for g, v in json.loads(p.read_text())["bits"].items()}
        T = float(json.loads(p.read_text())["avg_bit"])
        t0 = time.time()
        uA, uB = U_split(bits)
        print(f"[cond45-refine] start avg={avg(bits):.4f} uA={uA:+.5f} uB={uB:+.5f}", flush=True)
        for r in range(1, 4):
            best = None
            for i, j in permutations(blocks_sorted, 2):
                if bits[i] >= BIT_HI or bits[j] <= BIT_LO:
                    continue
                cand = dict(bits)
                cand[i] += 1
                cand[j] -= 1
                if avg(cand) > T + 1e-9:
                    continue
                cA, cB = U_split(cand)
                dA, dB = cA - uA, cB - uB
                if dA + dB > 0 and min(dA, dB) >= -EPS:
                    score = dA + dB
                    if best is None or score > best[1]:
                        best = ((i, j, cA, cB), score)
            if best is None:
                print(f"  [cond45-refine r{r}] fixed point", flush=True)
                break
            (i, j, cA, cB), score = best
            bits[i] += 1
            bits[j] -= 1
            uA, uB = cA, cB
            print(f"  [cond45-refine r{r}] {i}+1/{j}-1 gain={score:+.5f} avg={avg(bits):.4f} ({time.time()-t0:.0f}s)", flush=True)
        eval5k(bits, "swins-cond45-refined", "cond45_refined.json")
    else:
        print("[cond45-refine] cond45_5k.json missing; skipping", flush=True)

    applier.restore()
    print("EXP63 DONE", flush=True)


if __name__ == "__main__":
    main()
