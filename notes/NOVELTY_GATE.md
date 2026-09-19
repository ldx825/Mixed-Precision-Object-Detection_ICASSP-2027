# Gate N — Novelty (Phase II)

Date: 2026-09-12. Basis: prior audits (`BLOBQ/LAMPQ/REGPTQ_METHOD_DERIVATION.md`),
`OUTCALSD_vs_PRIOR_EQUIVALENCE_AUDIT.md`, `PHASE2_EXP00_RESULT_AUDIT.md`, and two new analyses
run today (`rank_singleton_vs_conditional`, `additive_selection_test`).

## Gate N criteria check

### N1 — OUTCALsd ≉ BLOB-Q distortion, and rankings actually differ
- **Formula level: holds.** Different objects (structured detector outputs, measured, with
  context) vs classification-style L2, 2nd-order approximated, assumed additive.
- **Behavior level: holds via stand-in.** Singleton-ranking family (empirical stand-in for
  Γ_i/Fisher-style allocation) as a *decision policy* = 31.39 AP vs conditional = 34.95 AP
  (same budget, 5k). Reversals 25–45/66 across analyses.
- **Residual:** a strict BLOB-Q-style Γ reimplementation ranking check belongs to Gate B prep.

### N2 — Novelty = context-dependent / non-additive precision utility  → STRONG EVIDENCE
- S3 (same base, symmetric ops): Kendall **+0.242**, 25/66 reversals, top-4 overlap 2/4.
- Additive prediction error at K=4: **−4.95 … −6.12 AP** (interaction ≈ 40–50% of singleton sum).
- SEL_S1 (top-4 by singleton damage; **our empirical stand-in family for independent
  singleton rankings incl. Fisher-style scores — not the official pipeline of any cited
  method**) ≈ random (31.39 vs 31.07 RAND; probe512-comparable 31.90).
- This upgrades N2 from hypothesis to measured phenomenon (controlled setting, 5k-verified at
  key points).

### N3 — 64-sample GT-free estimator of conditional utility  → HOLDS
- OUTCALsd (calib64-only) = 34.95 vs ORAC 35.32 → recovers **97.9%** of Oracle improvement over BASE4.
- 6-bit: OUT directions 2/2 correct where FEAT 0/2.

### N4 — Optimization/theory structure differs from BLOB-Q DP / LampQ ILP
- **Not yet done** (post-Gate-B work).

## Verdict

```text
Gate N: CONDITIONAL GO — proceed to STEP 8 (freeze quantization contract) and STEP 9 (true same-budget 6/5-bit).
```

The novelty statement to carry forward (replacing "output-aware allocation", which is occupied):

> **Detector precision utility is context-dependent and non-additive; our singleton-metric
> stand-ins mis-allocate bits in our controlled experiment (near-chance decisions on this
> setting, 3.6 AP below measure-what-you-decide at identical budget), while a GT-free 64-image
> measurement of conditional marginal utility recovers 97.9% of the oracle improvement.**

## Mandatory risk register (carried into Gate B)

1. **BLOB-Q has detection/segmentation experiments** (Mask R-CNN + Swin-T/S on COCO, 4/5/6-bit —
   authoritative external information, 2026-09-12; not located in our retrieved supp copy, which
   is a retrieval limitation of this audit only). Our novelty therefore must NOT rest on
   "no prior detection validation" — it rests on the **conditional / non-additive** axes (N2)
   and their **GT-free measurement** (N3). Behavioral comparison vs a Γ-style additive
   stand-in at equal budget = part of Gate B.
2. **Fairness/alignment**: LampQ & BLOB-Q detector results use **W+A** quantization; ours currently
   **W-only**. The frozen contract must either add activation quantization or state the deviation
   explicitly in all comparisons (plan §12/§46).
3. **Strict Γ-reimplementation ranking check** (BLOB-Q-style additive ranking vs OUTCALsd) —
   schedule as part of Gate B; current proxy (SEL_S1) is empirical, must be labeled as such.
4. **Cross-paper numbers are not comparable** (FP 46.0/41.6 v2.x-era vs our 42.65 v3.3); all
   head-to-head must be same-checkpoint/quantizer/calib/budget.

## Evidence pointers

```
src/diagnostics/rank_singleton_vs_conditional.py   -> S1–S4 numbers
src/diagnostics/additive_selection_test.py         -> SEL_S1 = 31.39 (5k) / 31.90 (probe512)
outputs/Exp04_additive_test/metrics.json
notes/OUTCALSD_vs_PRIOR_EQUIVALENCE_AUDIT.md       -> master table + verdicts
```
