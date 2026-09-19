"""Exp31 — Context rank reversal analysis for |S| = 0,1,2,3.

Data: outputs/Exp30_pair_interaction/runs.json
      (pairs from Exp30; full triples + quadruples from Exp30b)
      singleton utilities reused from outputs/Exp04_budget6/UP_*/calib_churn.json
Utility: U(key) = -score_drop (higher = better agreement with FP teacher on calib64).

For each context S (|S| = 0,1,2,3), candidates C = blocks minus S:
  conditional:  Delta(i|S) = U(S∪{i}) - U(S)
  singleton:    Delta(i|0) = U({i}) - U(0)
Compare rankings: Kendall tau, Spearman, pairwise reversal rate (raw + thresholded),
top-1 mismatch, top-3 overlap. Aggregated per level + pooled + concrete reversal examples.

Outputs: outputs/Exp31_context_rank_reversal/{summary.json, contexts.csv, examples.json}
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
OUT = ROOT / "outputs/Exp31_context_rank_reversal"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = json.loads((ROOT / "outputs/Exp30_pair_interaction/runs.json").read_text())
BLOCKS = sorted(json.loads((ROOT / "outputs/Exp04_budget6" / "UP_list.json").read_text())
                if (ROOT / "outputs/Exp04_budget6/UP_list.json").exists()
                else [p.name.replace("UP_", "") for p in sorted((ROOT / "outputs/Exp04_budget6").glob("UP_*"))
                      if p.is_dir()])
assert len(BLOCKS) == 12, BLOCKS

SINGLES = {}
for g in BLOCKS:
    c = json.loads((ROOT / f"outputs/Exp04_budget6/UP_{g}/calib_churn.json").read_text())
    SINGLES[g] = -c["score_drop"]
U0 = -RUNS["BASE"]["score_drop"]
BASE_DELTA = {i: SINGLES[i] - U0 for i in BLOCKS}

REV_THR = 0.002  # score_drop units; "meaningful" reversal threshold


def U(key):
    key = tuple(sorted(key))
    if len(key) == 1:
        return SINGLES[key[0]]
    k = "+".join(key)
    if k not in RUNS:
        return None
    return -RUNS[k]["score_drop"]


def pair_reversals(a, b):
    """Count order reversals between two dicts of scores on shared keys."""
    ks = sorted(set(a) & set(b))
    tot = rev = rev_thr = 0
    mags = []
    for i, j in itertools.combinations(ks, 2):
        d = a[i] - a[j]
        e = b[i] - b[j]
        if d == 0 or e == 0:
            continue
        tot += 1
        if (d > 0) != (e > 0):
            rev += 1
            mags.append(min(abs(d), abs(e)))
            if abs(d) >= REV_THR and abs(e) >= REV_THR:
                rev_thr += 1
    return rev, tot, rev_thr, mags


def analyze(S):
    """S: tuple of context blocks. Returns metric dict or None if data missing."""
    uS = U(S) if S else U0
    if uS is None:
        return None
    cond, base = {}, {}
    for i in BLOCKS:
        if i in S:
            continue
        uSi = U(tuple(S) + (i,))
        if uSi is None:
            return None
        cond[i] = uSi - uS
        base[i] = BASE_DELTA[i]
    ks = sorted(cond)
    c_arr = np.array([cond[i] for i in ks])
    b_arr = np.array([base[i] for i in ks])
    tau = stats.kendalltau(c_arr, b_arr).statistic
    rho = stats.spearmanr(c_arr, b_arr).statistic
    rev, tot, rev_thr, mags = pair_reversals(base, cond)
    top1_c = max(ks, key=lambda i: cond[i])
    top1_b = max(ks, key=lambda i: base[i])
    top3_c = set(sorted(ks, key=lambda i: -cond[i])[:3])
    top3_b = set(sorted(ks, key=lambda i: -base[i])[:3])
    return {
        "S": list(S), "n_cand": len(ks),
        "kendall": float(tau), "spearman": float(rho),
        "rev": rev, "rev_tot": tot, "rev_thr": rev_thr, "rev_mags": mags,
        "top1_cond": top1_c, "top1_base": top1_b, "top1_match": top1_c == top1_b,
        "top3_overlap": len(top3_c & top3_b) / 3,
        "cond": cond, "base": base,
        "gain_top1": cond[top1_c] - cond.get(top1_b, 0.0),
    }


def main():
    all_rows = []
    examples = []
    levels = [0, 1, 2, 3]
    summary = {}
    for L in levels:
        rows = []
        for S in itertools.combinations(BLOCKS, L):
            r = analyze(S)
            if r is None:
                continue
            rows.append(r)
            all_rows.append((L, r))
        if not rows:
            summary[f"L{L}"] = {"n_contexts": 0}
            continue
        agg = {
            "n_contexts": len(rows),
            "kendall_mean": float(np.mean([r["kendall"] for r in rows])),
            "kendall_median": float(np.median([r["kendall"] for r in rows])),
            "spearman_mean": float(np.mean([r["spearman"] for r in rows])),
            "reversal_rate_pooled": float(sum(r["rev"] for r in rows) / max(sum(r["rev_tot"] for r in rows), 1)),
            "reversal_rate_thr_pooled": float(sum(r["rev_thr"] for r in rows) / max(sum(r["rev_tot"] for r in rows), 1)),
            "top1_mismatch_frac": float(np.mean([1 - r["top1_match"] for r in rows])),
            "top3_overlap_mean": float(np.mean([r["top3_overlap"] for r in rows])),
        }
        all_mags = [m for r in rows for m in r.get("rev_mags", [])]
        if all_mags:
            agg["reversal_mag_p50"] = float(np.median(all_mags))
            agg["reversal_mag_p90"] = float(np.percentile(all_mags, 90))
            agg["reversal_mag_mean"] = float(np.mean(all_mags))
        summary[f"L{L}"] = agg
        # collect dramatic examples: top1 mismatch with largest conditional gain of true top1
        for r in sorted(rows, key=lambda x: -abs(x["gain_top1"]))[:5]:
            if not r["top1_match"]:
                examples.append({"level": L, **{k: v for k, v in r.items() if k not in ("cond", "base")}})

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "examples.json").write_text(json.dumps(examples, indent=2))
    with open(OUT / "contexts.csv", "w") as f:
        f.write("level,S,kendall,spearman,rev,rev_tot,rev_thr,top1_cond,top1_base,top1_match,top3_overlap,gain_top1\n")
        for L, r in all_rows:
            f.write(f"{L},{'+'.join(r['S']) or 'EMPTY'},{r['kendall']:.4f},{r['spearman']:.4f},"
                    f"{r['rev']},{r['rev_tot']},{r['rev_thr']},{r['top1_cond']},{r['top1_base']},"
                    f"{int(r['top1_match'])},{r['top3_overlap']:.3f},{r['gain_top1']:.5f}\n")

    print("=== Context rank reversal summary ===")
    for L in levels:
        s = summary[f"L{L}"]
        if s.get("n_contexts", 0) == 0:
            print(f"  |S|={L}: (no data)")
            continue
        print(f"  |S|={L}: n={s['n_contexts']:3d}  Kendall(mean)={s['kendall_mean']:+.3f}  "
              f"reversal={s['reversal_rate_pooled']*100:.1f}% (thr {s['reversal_rate_thr_pooled']*100:.1f}%)  "
              f"top1-mismatch={s['top1_mismatch_frac']*100:.0f}%  top3-overlap={s['top3_overlap_mean']:.2f}")
    print("\n=== Examples (top1 mismatches, sorted by |gain|) ===")
    for e in examples[:8]:
        print(f"  |S|={e['level']} S={e['S']}: base-top1={e['top1_base']} cond-top1={e['top1_cond']} "
              f"(gain {e['gain_top1']:+.4f})")

    # figure: reversal rate by level
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ls = [L for L in levels if summary[f"L{L}"].get("n_contexts", 0) > 0]
    rev = [summary[f"L{L}"]["reversal_rate_pooled"] * 100 for L in ls]
    revt = [summary[f"L{L}"]["reversal_rate_thr_pooled"] * 100 for L in ls]
    x = np.arange(len(ls))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(x - 0.18, rev, 0.36, label="raw reversal %")
    ax.bar(x + 0.18, revt, 0.36, label=f"reversal % (|gap|>{REV_THR})")
    ax.set_xticks(x); ax.set_xticklabels([f"|S|={L}" for L in ls])
    ax.set_ylabel("pairwise order reversal rate")
    ax.legend(fontsize=8)
    ax.set_title("Context rank reversal vs singleton ranking (calib64, U6 base)")
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures/rank_reversal_summary.pdf")
    print("\nwritten outputs/Exp31_context_rank_reversal/*")
    print("EXP31 DONE")


if __name__ == "__main__":
    main()
