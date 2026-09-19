"""Exp30 — Pair interaction matrix (66 pairs) + sampled triples (20) on calib64.

Utility (GT-free, detector-output):
    U(S) = -score_drop(S vs FP teacher)   [primary]
    U_lost(S) = -lost_frac(S vs FP)        [secondary]
Base: U6 (all blocks 6-bit). Action: repair block(s) in S to 8-bit.
Interaction:  I_ij = U({i,j}) - U({i}) - U({j}).

Singleton U({i}) is reused from outputs/Exp04_budget6/UP_*/calib_churn.json
(same contract/base; computed with the identical `churn` code).
New inference: 66 pairs + 20 triples (hi-6 blocks) on 64 calib images.
Incremental save after every config -> resumable.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
sys.path.insert(0, str(ROOT / "src"))

from baseline.exp01_repro_fp import build_model  # noqa: E402
from diagnostics.sweep_v2 import get_block_and_stage_groups  # noqa: E402
from quant.fake_quant import GroupQuantApplier, quantize_weight_tensor  # noqa: E402
from diagnostics.confirm_5k import run_calib_dets  # noqa: E402
from diagnostics.actionability_probe import churn  # noqa: E402

HI6 = ["BLK2.2", "BLK2.3", "BLK2.4", "BLK2.5", "BLK3.0", "BLK3.1"]


def main():
    out_root = ROOT / "outputs/Exp30_pair_interaction"
    out_root.mkdir(parents=True, exist_ok=True)
    runs_file = out_root / "runs.json"
    runs = json.loads(runs_file.read_text()) if runs_file.exists() else {}

    model, cfg = build_model()
    applier = GroupQuantApplier(model)
    blocks, stg = get_block_and_stage_groups(model)
    pd = dict(model.named_parameters())
    all_names = {**blocks, **stg}
    blocks_sorted = sorted(blocks)

    def set_state(bits_map):
        applier.restore()
        for gname, bits in bits_map.items():
            for n in all_names[gname]:
                pd[n].data.copy_(quantize_weight_tensor(pd[n].data, bits))
        torch.cuda.synchronize()

    from pycocotools.coco import COCO

    coco = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
    cat_ids = coco.getCatIds()

    # FP teacher detections on calib64
    applier.restore()
    fp_dets = run_calib_dets(model, cat_ids)

    # U6 base utility
    if "BASE" not in runs:
        set_state({g: 6 for g in blocks})
        base_dets = run_calib_dets(model, cat_ids)
        ch = churn(fp_dets, base_dets)
        runs["BASE"] = ch
        runs_file.write_text(json.dumps(runs, indent=2))
        print(f"[BASE] score_drop={ch['score_drop']:+.4f} lost={ch['lost_frac']:.4f}", flush=True)

    pairs = list(itertools.combinations(blocks_sorted, 2))
    triples = list(itertools.combinations(HI6, 3))

    def run_combo(combo):
        key = "+".join(combo)
        if key in runs:
            return
        t0 = time.time()
        bits_map = {**{g: 6 for g in blocks}, **{g: 8 for g in combo}}
        set_state(bits_map)
        dets = run_calib_dets(model, cat_ids)
        ch = churn(fp_dets, dets)
        runs[key] = ch
        runs_file.write_text(json.dumps(runs, indent=2))
        print(f"[{key}] score_drop={ch['score_drop']:+.4f} lost={ch['lost_frac']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    for c in pairs:
        run_combo(c)
    for c in triples:
        run_combo(c)
    applier.restore()

    # ---------------- analysis ----------------
    def U(key):  # primary: -score_drop
        return -runs[key]["score_drop"]

    def U_lost(key):
        return -runs[key]["lost_frac"]

    # singleton utilities (reuse budget6 UP_*; fall back to rerun if missing)
    sing = {}
    sing_lost = {}
    for g in blocks_sorted:
        p = ROOT / f"outputs/Exp04_budget6/UP_{g}/calib_churn.json"
        c = json.loads(p.read_text())
        sing[g] = -c["score_drop"]
        sing_lost[g] = -c["lost_frac"]

    I = {}
    for (i, j) in pairs:
        I[f"{i}|{j}"] = U(f"{i}+{j}") - sing[i] - sing[j]
    I_vals = np.array(list(I.values()))
    summary = {
        "n_pairs": len(I),
        "mean_abs_I": float(np.mean(np.abs(I_vals))),
        "median_abs_I": float(np.median(np.abs(I_vals))),
        "p90_abs_I": float(np.percentile(np.abs(I_vals), 90)),
        "max_abs_I": float(np.max(np.abs(I_vals))),
        "pos_frac": float(np.mean(I_vals > 0)),
        "neg_frac": float(np.mean(I_vals < 0)),
        "mean_abs_singleton": float(np.mean(np.abs(list(sing.values())))),
        "interaction_over_singleton": float(np.mean(np.abs(I_vals)) / (np.mean(np.abs(list(sing.values()))) + 1e-12)),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_root / "interactions.json").write_text(json.dumps(I, indent=2))

    # matrix + csv
    idx = {g: n for n, g in enumerate(blocks_sorted)}
    M = np.zeros((12, 12))
    for (i, j) in pairs:
        M[idx[i], idx[j]] = M[idx[j], idx[i]] = I[f"{i}|{j}"]
    np.save(out_root / "interaction_matrix.npy", M)
    with open(out_root / "pairs.csv", "w") as f:
        f.write("i,j,I_ij,U_ij,U_i,U_j\n")
        for (i, j) in pairs:
            f.write(f"{i},{j},{I[f'{i}|{j}']:.6f},{U(f'{i}+{j}'):.6f},{sing[i]:.6f},{sing[j]:.6f}\n")

    print("\n=== interaction summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M, cmap="RdBu_r")
    ax.set_xticks(range(12)); ax.set_xticklabels(blocks_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(12)); ax.set_yticklabels(blocks_sorted, fontsize=8)
    ax.set_title("Pair interaction I_ij = U(i,j) - U(i) - U(j)  (U6 base, calib64, -score_drop)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures/pair_interaction_heatmap.pdf")
    fig.savefig(ROOT / "figures/pair_interaction_heatmap.png", dpi=160)
    print("\nwritten outputs/Exp30_pair_interaction/* + figures/pair_interaction_heatmap.*")
    print("EXP30 DONE")


if __name__ == "__main__":
    main()
