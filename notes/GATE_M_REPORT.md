# Gate M — Mechanism Decision Report

Date: 2026-09-12 (Phase II, STEP 7). Inputs: Exp30 (66 pairs), Exp30b (220 triples + 495 quads),
Exp31 (context rank reversal |S|=0–3), Exp33 (K=4 decision comparison), Phase-I artifacts
(SEL_S1/OUT/ORAC 5k results, swap tests). Utility: U(S) = −score_drop(state S; FP teacher) on
calib64, U6 base (all blocks 6-bit; repaired S → 8-bit). GT-free throughout.

Two interaction definitions are reported:
- **I_raw** (Addendum §8, literal): I = U({i,j}) − U({i}) − U({j});
- **I_cent** (decision-relevant, centered): I = U({i,j}) − U({i}) − U({j}) + U(∅),
  the correct decomposition term since U(∅) = −0.0298 ≠ 0 (I_raw is dominated by the constant
  |U(∅)| and is 100% positive by construction; all judgments below use I_cent).

## M1 — Pair interactions are not numerically negligible  → **GO**

66 pairs (full matrix, `outputs/Exp30_pair_interaction/`):

| statistic | value |
|---|---|
| mean I_cent | +0.00048 (near-zero mean → no systematic super/sub-additivity) |
| std I_cent | 0.00174 |
| range | −0.00470 … +0.00597 |
| positive / negative | 58% / 42% |
| mean \|I_cent\| | **0.00135** |
| mean \|Δ_i\| (singleton effect) | 0.00171 |
| **ratio mean\|I\| / mean\|Δ\|** | **0.79** |

**Cross-block interactions are ~79% of the size of singleton effects on average** — the additive
(singleton-sum) model mis-states joint utility by an amount comparable to the quantities being
summed.  Examples of the extremes:

- `BLK2.3 + BLK2.5`: I = **+0.0060** — each block is nearly useless alone (Δ=+0.0008 / +0.0000),
  jointly they give the largest recorded gain;
- `BLK2.0 + BLK2.2`: I = **−0.0047** — each is among the best singletons (+0.0031/+0.0032), jointly
  they under-perform their sum by the largest negative amount;
- `BLK3.0 + BLK3.1`: I = +0.0044; `BLK1.0 + BLK3.0`: I = −0.0038.

## M2 — Context rank reversal is common and affects allocation  → **GO**

`outputs/Exp31_context_rank_reversal/` (conditional ranking Δ(i|S) vs singleton ranking Δ(i|∅);
candidates exclude blocks already in S):

| level | #contexts | Kendall τ (mean) | Spearman | **pairwise reversal rate** | top-1 mismatch | top-3 overlap | reversal mag p50 / p90 | thr-reversal* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| \|S\|=1 | 12 | +0.548 | +0.674 | **22.6%** | 25% | 0.81 | 0.00054 / 0.00164 | 1.7% |
| \|S\|=2 | 66 | +0.506 | +0.648 | **24.7%** | 61% | 0.60 | 0.00060 / 0.00229 | 3.4% |
| \|S\|=3 | 220 | +0.472 | +0.604 | **26.4%** | 62% | 0.60 | 0.00060 / 0.00229 | 3.7% |

*thr-reversal = both compared gaps ≥ 0.002 (conservative "large-reversal" count).

**Reading (honest):** the ≥20% Addendum bar is met at every level; reversal frequency grows with
context depth while Kendall degrades (0.55→0.47). Most reversals are small in gap size
(p50 ≈ 0.0006), but decision-relevant instability is substantial: top-1 mismatch 25% (|S|=1) to
62% (|S|=2–3), top-3 overlap drops to 0.60. Large-gap reversals exist and can dominate:
`S={BLK2.5, BLK3.0}` — singleton ranking ⇒ repair BLK2.4; conditional measurement ⇒ repair
BLK2.3 instead, with a **+0.0070 utility swing (≈4× the mean singleton effect)**;
`S={BLK0.1, BLK2.3}` — +0.0040. (Note: as |S| grows, part of the top-1 mismatch is because the
strongest singletons are already repaired; the decision-relevant statement is that the static
ranking path and the conditional-optimal path diverge frequently.)

## M3 — Independent ranking causes measurable same-budget AP loss  → **GO**

**Decision-level, equal repair budget (4/12 blocks → 8-bit), 5k box AP:**

| policy | 5k AP |
|---|---:|
| singleton-ranking-family decision (SEL_S1; stand-in) | 31.39 |
| RANDOM allocation (probe512-comparable) | 31.07 |
| **conditional OUT (ours)** | **34.95** |
| ORAC | 35.32 |
| BASE4 (no repair) | 17.67 |

→ independent ranking loses **−3.56 AP** vs the conditional decision at identical budget
(6-bit swap test: OUT directions 2/2 correct, FEAT 0/2).

**Utility-level check (Exp33, exact enumeration of all 495 K=4 configurations):**

| policy | set | U | rank / 495 |
|---|---|---:|---:|
| singleton top-4 | BLK2.0, BLK2.2, BLK2.4, BLK3.0 | −0.01749 | 41 |
| conditional greedy (exact lookups) | BLK2.3, BLK2.4, BLK3.0, BLK3.1 | −0.01479 | **11** |
| oracle | BLK2.2, BLK2.3, BLK2.4, BLK3.0 | −0.01238 | 1 |

Gaps: singleton→oracle +0.00510, greedy→oracle +0.00241 (**greedy recovers 53% of the gap**),
singleton→greedy +0.00269. One block swapped (BLK2.0→BLK2.3) moves the set from rank 41 to 1 —
the decision surface is genuinely context-sensitive.

## M4 — The 64-image conditional estimator predicts the better action  → **GO**

- **End-to-end (5k, Phase I):** OUTCALsd (calib64-only, GT-free) = 34.95 vs ORAC = 35.32 →
  recovers **97.9%** of the oracle gain over BASE4; OUT-512 = 34.42.
- **Perfect-measurement upper bound (Exp33):** conditional greedy with exact table lookups ranks
  **11/495** (z = +2.28 vs random-quad distribution −0.0223 ± 0.0033) — i.e. the conditional
  *structure* itself, not estimator noise, delivers most of the improvement; the 64-image
  estimator achieves 97.9% of oracle, so estimation noise costs little.
- **Seed robustness:** Exp34 (5 seeds × n = 8…256, per-image recomputable) running — will be
  reported under Addendum §19 before Gate B conclusions.

## Verdict

```text
[Gate M]

Question:        Is the mechanism claim (context-dependent, non-additive precision utility with
                 actionable rank reversal) empirically supported?
Exact evidence:  Exp30 (interaction matrix; mean|I|/mean|Delta| = 0.79), Exp31 (reversal 22.6–26.4%,
                 top-1 mismatch 25–62%), Exp33 (rank 41 vs 11 vs 1 of 495), Phase-I 5k decisions.
Numbers:         above tables; key: +0.0060 / −0.0047 pair interactions, 22.6–26.4% reversals,
                 −3.56 AP same-budget loss for independent ranking, 97.9% oracle recovery.
Comparison to strongest prior: behavioral comparison vs Gamma-style additive stand-ins at equal
                 budget (31.39 vs 34.95); true-budget comparison = Gate B (next).
What claim is now allowed?    M1–M4 as stated (with the small-reversal-magnitude caveat).
What claim is still forbidden? "BLOB-Q has no detection experiment"; "LampQ is approximately random"
                 (stand-in wording only); presenting +12.2 AP as the final improvement; claiming
                 large reversals dominate (most are ~6e-4; large ones exist but are the minority).
Decision:        GO — proceed to STEP 8/9 (true same-budget 6.0 / 5.0 / 4.5 head-to-head).
Next experiment: Gate B: Uniform / F-LBQ-style / BLOB-Q-style-reimpl / LampQ-style stand-in /
                 Static OUT singleton / Conditional OUT (greedy) / Oracle at matched budgets,
                 under the frozen QUANTIZATION_CONTRACT; then calibration-size + seed robustness
                 report (Exp34), theory path (Addendum §18 A: sample complexity), second backbone.
```

## Caveats carried into Gate B

1. All mechanism numbers are on one calibration set (calib64 seed0); Exp34 seed robustness is the
   required check (§19) and is running.
2. Utility = score_drop-based; the AP-level basis for M3 comes from the Phase-I decisions
   (5k-verified) under the same contract.
3. I_raw vs I_cent distinction must appear in any write-up (never quote I_raw = +0.0303 as
   "interaction size").
4. Everything GT-free; detections-only matching (IoU 0.5 / score 0.3), boxes only.

## Artifacts

```
outputs/Exp30_pair_interaction/{runs.json, pairs.csv, interaction_matrix.npy, summary.json}
figures/pair_interaction_heatmap.{pdf,png}
outputs/Exp31_context_rank_reversal/{summary.json, contexts.csv, examples.json, decision_k4.json}
figures/rank_reversal_summary.pdf
outputs/Exp34_seed_curve/ (seed robustness, running)
```
