"""Exp35 analysis only (CPU, no GPU): cross-seed validation of pair interactions + |S|=1 reversal.

Reads outputs/Exp35_seed_pairs/seed{1..4}_runs.json (already collected) + Exp30/budget6 for seed0.
Fixes the 'P_' key-prefix bug in the original in-script analysis.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
OUT = ROOT / "outputs/Exp35_seed_pairs"
SEEDS = [1, 2, 3, 4]

per_seed = {}
for seed in SEEDS:
    runs = json.loads((OUT / f"seed{seed}_runs.json").read_text())
    blocks = sorted(k[3:] for k in runs if k.startswith("UP_"))
    u0 = -runs["U6"]["churn"]["score_drop"]
    sing = {g: -runs[f"UP_{g}"]["churn"]["score_drop"] for g in blocks}
    delta = {g: sing[g] - u0 for g in blocks}
    I = {}
    for (i, j) in itertools.combinations(blocks, 2):
        uij = -runs[f"P_{i}+{j}"]["churn"]["score_drop"]
        I[f"{i}+{j}"] = uij - sing[i] - sing[j] + u0
    # |S|=1 reversal
    revs = []
    for j in blocks:
        cond, base = {}, {}
        for i in blocks:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            uij = -runs[f"P_{a}+{b}"]["churn"]["score_drop"]
            cond[i] = uij - sing[j]
            base[i] = delta[i]
        ks = sorted(cond)
        tot = rev = 0
        for a, b in itertools.combinations(ks, 2):
            d = cond[a] - cond[b]
            e = base[a] - base[b]
            if d == 0 or e == 0:
                continue
            tot += 1
            if (d > 0) != (e > 0):
                rev += 1
        revs.append(rev / max(tot, 1))
    per_seed[seed] = {"delta": delta, "I": I, "reversal": float(np.mean(revs)),
                      "kendall_vs_singleton": float(np.mean([stats.kendalltau(
                          [cond[i] for i in sorted(cond)], [base[i] for i in sorted(cond)]).statistic
                          for cond, base in []])) if False else None}
    # per-context kendall mean (recompute cleanly)
    kts = []
    for j in blocks:
        cond, base = {}, {}
        for i in blocks:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            uij = -runs[f"P_{a}+{b}"]["churn"]["score_drop"]
            cond[i] = uij - sing[j]
            base[i] = delta[i]
        ks = sorted(cond)
        kts.append(stats.kendalltau([cond[i] for i in ks], [base[i] for i in ks]).statistic)
    per_seed[seed]["kendall_mean"] = float(np.mean(kts))
    print(f"seed{seed}: mean|I|={np.mean(np.abs(list(I.values()))):.5f} "
          f"reversal={per_seed[seed]['reversal']*100:.1f}% kendall={per_seed[seed]['kendall_mean']:+.3f}",
          flush=True)

# seed0 from Exp30 (no prefix)
s0 = json.loads((ROOT / "outputs/Exp30_pair_interaction/runs.json").read_text())
blk0 = sorted(p.name.replace("UP_", "") for p in (ROOT / "outputs/Exp04_budget6").glob("UP_*") if p.is_dir())
sing0 = {g: -json.loads((ROOT / f"outputs/Exp04_budget6/UP_{g}/calib_churn.json").read_text())["score_drop"]
         for g in blk0}
u00 = -s0["BASE"]["score_drop"]
I0 = {}
for (i, j) in itertools.combinations(blk0, 2):
    I0[f"{i}+{j}"] = -s0[f"{i}+{j}"]["score_drop"] - sing0[i] - sing0[j] + u00
delta0 = {g: sing0[g] - u00 for g in blk0}
per_seed[0] = {"delta": delta0, "I": I0}

# cross-seed correlation matrices (Pearson + Kendall on the 66 interaction values)
keys = sorted(I0)
assert all(sorted(per_seed[s]["I"]) == keys for s in per_seed)
mat = np.array([[per_seed[s]["I"][k] for k in keys] for s in sorted(per_seed)])
corr = np.corrcoef(mat)
kt = np.array([[stats.kendalltau(mat[a], mat[b]).statistic for b in range(len(mat))] for a in range(len(mat))])

print("\n=== Pearson corr of I-vectors (rows/cols: seeds 0..4) ===")
for a in range(len(mat)):
    print("  " + " ".join(f"{corr[a][b]:+.2f}" for b in range(len(mat))))
print("\n=== Kendall tau of I-vectors ===")
for a in range(len(mat)):
    print("  " + " ".join(f"{kt[a][b]:+.2f}" for b in range(len(mat))))

print("\n=== ratio mean|I|/mean|Delta| per seed ===")
for s in sorted(per_seed):
    r = np.mean(np.abs(list(per_seed[s]["I"].values()))) / np.mean(np.abs(list(per_seed[s]["delta"].values())))
    print(f"  seed{s}: {r:.2f}")

print("\n=== |S|=1 reversal + kendall per seed (0 = Exp30/calib64_seed0) ===")
for s in sorted(per_seed):
    if "reversal" in per_seed[s]:
        print(f"  seed{s}: reversal={per_seed[s]['reversal']*100:.1f}%  kendall={per_seed[s]['kendall_mean']:+.3f}")

summary = {
    "I_pearson": corr.tolist(),
    "I_kendall": kt.tolist(),
    "per_seed": {str(s): {"ratio_I_over_delta": float(np.mean(np.abs(list(per_seed[s]["I"].values()))) /
                                                        np.mean(np.abs(list(per_seed[s]["delta"].values())))),
                          "reversal": per_seed[s].get("reversal"),
                          "kendall_mean": per_seed[s].get("kendall_mean")}
                 for s in sorted(per_seed)},
}
(OUT / "seed_validation_summary.json").write_text(json.dumps(summary, indent=2))
print("\nwritten seed_validation_summary.json")
print("EXP35 ANALYSIS DONE")
