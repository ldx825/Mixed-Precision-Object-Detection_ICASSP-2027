"""Exp33 — Decision-level comparison at K=4 repair (Addendum §15/§16, mechanism scale).

Input: full data in outputs/Exp30_pair_interaction/runs.json
       (BASE + 66 pairs from Exp30; 220 triples + 495 quads from Exp30b)
       singletons from outputs/Exp04_budget6/UP_*/calib_churn.json
Utility: U(S) = -score_drop(S; FP) on calib64 (U6 base, repair to 8-bit).
All policies ranked on the SAME 495-candidate quad space (exact enumeration = tractable).

Policies:
  P1  singleton top-4  (Delta_i = U({i}) - U0; one-shot, the deployable-family stand-in)
  P2  conditional greedy (argmax Delta(i|S_t) via exact table lookups; the "perfect
      conditional measurement" limit — Gate-M decision structure)
  P3  exact optimum (argmax U over all C(12,4)=495 quads)
  P4  random-quad baseline (mean/std over all quads)

Reports utility rank of each policy's set (1=best of 495), utility gaps, set overlaps,
greedy path. Utility-level only (AP comparison belongs to Gate B).
"""
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")
OUT = ROOT / "outputs/Exp31_context_rank_reversal"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = json.loads((ROOT / "outputs/Exp30_pair_interaction/runs.json").read_text())
BLOCKS = sorted(p.name.replace("UP_", "") for p in (ROOT / "outputs/Exp04_budget6").glob("UP_*") if p.is_dir())
SING = {g: -json.loads((ROOT / f"outputs/Exp04_budget6/UP_{g}/calib_churn.json").read_text())["score_drop"] for g in BLOCKS}
U0 = -RUNS["BASE"]["score_drop"]


def U(t):
    t = tuple(sorted(t))
    if len(t) == 0:
        return U0
    if len(t) == 1:
        return SING[t[0]]
    key = "+".join(t)
    if key not in RUNS:
        return None
    return -RUNS[key]["score_drop"]


def main():
    quads = list(itertools.combinations(BLOCKS, 4))
    uq = {q: U(q) for q in quads}
    missing = [q for q, v in uq.items() if v is None]
    uq = {q: v for q, v in uq.items() if v is not None}
    all_u = np.array(list(uq.values()))
    best_u = max(uq.values())

    def rank_of(S):
        uS = U(S)
        better = sum(1 for v in all_u if v > uS + 1e-12)
        return 1 + better, uS

    # P1: singleton top-4
    order = sorted(BLOCKS, key=lambda i: -(SING[i] - U0))
    P1 = tuple(sorted(order[:4]))
    # P2: conditional greedy (exact table lookup per step)
    S, path = (), []
    for _ in range(4):
        best_i, best_d = None, -1e18
        for i in BLOCKS:
            if i in S:
                continue
            d = U(S + (i,)) - U(S)
            if d > best_d:
                best_d, best_i = d, i
        S = S + (best_i,)
        path.append({"block": best_i, "delta": float(best_d)})
    P2 = tuple(sorted(S))
    # P3: exact optimum
    P3 = max(uq, key=lambda q: uq[q])

    r1, u1 = rank_of(P1)
    r2, u2 = rank_of(P2)
    r3, u3 = rank_of(P3)
    res = {
        "n_quads": len(uq),
        "missing_quads": len(missing),
        "singleton_order": order,
        "singleton_top4": list(P1),
        "singleton_top4_rank": r1,
        "singleton_top4_U": u1,
        "greedy_path": path,
        "greedy_set": list(P2),
        "greedy_rank": r2,
        "greedy_U": u2,
        "oracle_set": list(P3),
        "oracle_rank": 1,
        "oracle_U": u3,
        "random_mean_U": float(all_u.mean()),
        "random_std_U": float(all_u.std()),
        "gap_singleton_to_oracle": float(u3 - u1),
        "gap_greedy_to_oracle": float(u3 - u2),
        "gap_singleton_to_greedy": float(u2 - u1),
        "overlap_singleton_greedy": len(set(P1) & set(P2)),
        "overlap_greedy_oracle": len(set(P2) & set(P3)),
        "overlap_singleton_oracle": len(set(P1) & set(P3)),
        "z_singleton_vs_random": float((u1 - all_u.mean()) / (all_u.std() + 1e-12)),
        "z_greedy_vs_random": float((u2 - all_u.mean()) / (all_u.std() + 1e-12)),
    }
    (OUT / "decision_k4.json").write_text(json.dumps(res, indent=2))

    print("=== K=4 repair policy comparison (calib64 utility scale, 495 quads) ===")
    print(f"  random quad: U = {all_u.mean():+.5f} ± {all_u.std():.5f}")
    print(f"  P1 singleton top-4 : {list(P1)}  U={u1:+.5f}  rank {r1}/{len(uq)}  z={res['z_singleton_vs_random']:+.2f}")
    print(f"  P2 cond. greedy    : {list(P2)}  U={u2:+.5f}  rank {r2}/{len(uq)}  z={res['z_greedy_vs_random']:+.2f}")
    print(f"  P3 oracle          : {list(P3)}  U={u3:+.5f}  rank 1")
    print(f"  gaps: singleton->oracle {res['gap_singleton_to_oracle']:+.5f} | "
          f"greedy->oracle {res['gap_greedy_to_oracle']:+.5f} | singleton->greedy {res['gap_singleton_to_greedy']:+.5f}")
    print(f"  overlaps: P1∩P2={res['overlap_singleton_greedy']}/4  P2∩P3={res['overlap_greedy_oracle']}/4  P1∩P3={res['overlap_singleton_oracle']}/4")
    print(f"  greedy path: {[(p['block'], round(p['delta'],5)) for p in path]}")
    print("EXP33 DONE")


if __name__ == "__main__":
    main()
