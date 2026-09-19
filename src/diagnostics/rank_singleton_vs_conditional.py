"""Killer novelty analysis: singleton sensitivity ranking vs conditional marginal utility.

QUESTION: can any *independent* (singleton) sensitivity — the family of scores used by
F-LBQ (feature alteration), BLOB-Q (additive Γ), LampQ (Fisher) — predict the *conditional*
marginal utility measured inside a quantized context?

Uses existing artifacts (no GPU):
  4-bit world:  singleton = block-only damage (FP base, {b}->4bit)
                conditional = leave-up benefit (BASE4 base, {g} repaired to 8bit)
  6-bit world:  singleton = DOWN damage (U6 base, {g}->4bit)
                conditional = UP benefit (U6 base, {g}->8bit)
Report Spearman / Kendall / reversal count / top-4 overlap; also D_feat as singleton.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

R = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")


def report(name, x, y, labels):
    xs = np.array([x[b] for b in labels], float)
    ys = np.array([y[b] for b in labels], float)
    sp = stats.spearmanr(xs, ys)
    kt = stats.kendalltau(xs, ys)
    rev = sum(1 for i in range(len(labels)) for j in range(i + 1, len(labels))
              if (xs[i] - xs[j]) * (ys[i] - ys[j]) < 0)
    topx = set(sorted(labels, key=lambda b: -x[b])[:4])
    topy = set(sorted(labels, key=lambda b: -y[b])[:4])
    print(f"{name}")
    print(f"    Spearman={sp.statistic:+.3f} (p={sp.pvalue:.3f})  Kendall={kt.statistic:+.3f}  "
          f"reversals={rev}/{len(labels)*(len(labels)-1)//2}  top4overlap={len(topx & topy)}/4")
    print(f"    top4_singleton={sorted(topx)}")
    print(f"    top4_conditional={sorted(topy)}")
    return float(sp.statistic), float(kt.statistic), rev


def main():
    summ = json.loads((R / "tables/Exp03v2_all_summary.json").read_text())
    AP = {"FP": summ["AP_FP_probe512"], "BASE4": summ["AP_base4"]}

    print("=" * 70)
    print("4-BIT WORLD (probe512)")
    print("=" * 70)
    dfeat4, benefit4, bonly4 = {}, {}, {}
    for r in summ["designs"]["leave-up"]["rows"]:
        dfeat4[r["group"]] = r["D"]
        benefit4[r["group"]] = r["AP"] - AP["BASE4"]
    for r in summ["designs"]["block-only"]["rows"]:
        bonly4[r["group"]] = AP["FP"] - r["AP"]
    blk4 = sorted(benefit4)
    report("[S1] singleton damage (FP-base, block->4bit)  vs  conditional repair benefit (BASE4-base)",
           bonly4, benefit4, blk4)
    report("[S2] singleton D_feat (F-LBQ score)          vs  conditional repair benefit (BASE4-base)",
           dfeat4, benefit4, blk4)

    print()
    print("=" * 70)
    print("6-BIT WORLD (probe512, U6 base)")
    print("=" * 70)
    b6 = json.loads((R / "outputs/Exp04_budget6/budget6_summary.json").read_text())
    u6 = b6["U6_AP"]
    sb = b6["single_block"]
    blocks = sorted({k.split("_", 1)[1] for k in sb})
    down = {b: u6 - sb[f"DOWN_{b}"] for b in blocks}   # singleton damage at 6bit
    up = {b: sb[f"UP_{b}"] - u6 for b in blocks}       # conditional benefit at 6bit
    report("[S3] singleton damage ({g}->4bit from U6)   vs  conditional benefit ({g}->8bit from U6)",
           down, up, blocks)
    # singleton churn (calib) from budget6 DOWN runs
    churn_dn = {}
    for b in blocks:
        c = json.loads((R / f"outputs/Exp04_budget6/DOWN_{b}/calib_churn.json").read_text())
        churn_dn[b] = -c["score_drop"]  # score recovery if we "undo" the down-quant (analog)
    report("[S4] singleton calib score-recovery (DOWN)  vs  conditional benefit (UP)",
           churn_dn, up, blocks)

    print()
    print("=== summary of answers ===")
    print("If singleton rankings correlate STRONGLY with conditional rankings (Kendall>0.7, few reversals)")
    print("-> independent sensitivity suffices -> our non-additivity novelty WEAK.")
    print("If weak / reversals many -> independent priors mis-rank under context -> NOVELTY LIVES.")


if __name__ == "__main__":
    main()
