# OUTCALsd vs Prior — Equivalence Audit (Phase II, Gate N support)

Sources: `BLOBQ_METHOD_DERIVATION.md`, `LAMPQ_METHOD_DERIVATION.md`, `REGPTQ_METHOD_DERIVATION.md`,
`OUTCALSD_FORMULATION.md`; evidence artifacts in `notes/PHASE2_EXP00_RESULT_AUDIT.md`.

## 1. Master table

| Dimension | F-LBQ | BLOB-Q | LampQ | Reg-PTQ | OUTCALsd |
|---|---|---|---|---|---|
| Task | detection (Swin-T MRCNN) | classification + detection (COCO; Mask R-CNN Swin-T/S, 4/5/6-bit — authoritative info) | cls + detection (COCO) | detection | detection |
| PTQ/QAT | PTQ | PTQ | PTQ | PTQ | PTQ |
| Mixed precision | ✓ (backbone layers) | ✓ (layers) | ✓ (layer-wise) | ✗ (uniform INT4) | ✓ (12 blocks) |
| Calibration samples | 64 (train, random) | standard PTQ calib (n/split not public) | 32 (cls config; det. not public) | unstated in scan | **64 unlabeled** |
| Uses GT | No | No | likely (Fisher loss needs supervision; unverified) | Yes (global loss filtering; pseudo-box not found) | **No (zero GT anywhere)** |
| Sensitivity variable | feature alteration (backbone feature MSE) | per-layer 2nd-order distortion Γ_i (model-output Hessian/Jacobian) | type-aware Fisher score | — (no allocation) | **conditional marginal score-recovery S_g** |
| Local feature | ✓ (backbone) | ✗ (output-level via Hessian quadratic) | partial (gradient) | ✗ | ✗ (detector outputs) |
| Final output | ✗ | model output L2 (classification-style) | ✗ | global loss | **detector outputs (matched scores/lost/box)** |
| Jacobian/Hessian/Fisher | ✗ | ✓ (2nd-order) | ✓ (Fisher) | ✗ | ✗ (**direct measurement**) |
| Context-dependent | ✗ (singleton) | ✗ (additivity assumed) | ✗ (singleton) | n/a | **✓ (core property)** |
| Pair interaction | not modeled | assumed ≈0 (Finding 1) | not modeled (ILP additive) | n/a | **measured; non-negligible (see §2)** |
| Additive/decomposable | approx (LG on per-layer curves) | **✓ = central theorem** | ✓ (ILP formulation) | n/a | ✗ (**deliberately non-additive**) |
| Solver | Lagrangian multiplier search | DP (linear time, claims global opt under assumption) | ILP + iterative update | n/a | greedy conditional selection (current) |
| Object detection | ✓ | ✓ (Mask R-CNN Swin-T/S, COCO, 4/5/6-bit) | ✓ | ✓ | ✓ |
| Mask R-CNN + Swin | ✓ (Swin-T) | ✓ (Swin-T/S, 4/5/6-bit) | ✓ (Swin-T/S, 4MP/4MP) | ✗ (CNN RCNNs) | ✓ (Swin-T) |
| Main theorem | none (engineering) | Second-Order Additivity | none (empirical) | none | (planned: sample complexity / conditional-utility structure) |

## 2. Evidence on ranking equivalence (the decision-relevant part)

All numbers from `src/diagnostics/rank_singleton_vs_conditional.py` +
`src/diagnostics/additive_selection_test.py` + `notes/REGPTQ…` artifacts:

**(a) Singleton → conditional ranking agreement (12 blocks):**

| comparison | Spearman | Kendall | reversals | top-4 overlap |
|---|---:|---:|---:|---:|
| S1: singleton damage (FP-base) → cond. repair benefit (BASE4) | +0.622 | +0.424 | 19/66 | 2/4 |
| S2: D_feat (F-LBQ score) → cond. benefit | −0.441 | −0.273 | 42/66 | 0/4 |
| **S3: same base (U6), {g}→4bit damage → {g}→8bit benefit** | +0.364 | **+0.242** | 25/66 | 2/4 |
| S4: singleton calib score-recovery → cond. benefit | −0.517 | −0.364 | 45/66 | 0/4 |

**(b) Additive prediction error for K=4 combos (5k, AP points):**

| combo | additive pred | actual | interaction |
|---|---:|---:|---:|
| FEAT | 21.97 | 22.76 | +0.79 |
| OUT512 | 39.37 | 34.42 | **−4.95** |
| OUTCALsd | 41.07 | 34.95 | **−6.12** |
| ORAC | 41.22 | 35.32 | **−5.90** |

**(c) Decision-level head-to-head (same budget, repair 4/12, 5k box AP):**

| policy | 5k AP |
|---|---:|
| BASE4 | 17.67 |
| FEAT (D_feat top-4) | 22.76 |
| **SEL_S1 = top-4 by singleton damage (family stand-in for any singleton ranking)** | **31.39** |
| RAND | 31.07 (probe512; SEL_S1 probe512 = 31.90 → comparable) |
| OUT512 (conditional) | 34.42 |
| OUTCALsd (conditional, deployable) | 34.95 |
| ORAC | 35.32 |

## 3. The mandated question

> Is OUTCALsd merely a direct/low-order/empirical approximation of BLOB-Q's model-output distortion?

**Answer: Meaningfully Different** — on three axes:

1. **Object of distortion**: BLOB-Q approximates a *model-output L2* via 2nd-order expansion and
   *assumes* layer-uncorrelatedness (their Finding 1) to make it additive (detector-side
   formulation not audited from primary source; the retrieved derivation is classification-side).
   OUTCALsd *measures* the *detector's structured outputs* (matched-score recovery, lost detections,
   IoU) on a *quantized context*, with no additivity assumption. The quantities being estimated are
   different objects; one is not a truncation of the other.
2. **Context**: BLOB-Q's Γ_i are singleton quantities; our S3/S4 + §2(b) show singleton scores and
   conditional utilities disagree (Kendall 0.24–0.42; additive error 5–6 AP at K=4). A singleton
   estimator, however accurate, does not produce our decision variable.
3. **Failure regime**: the retrieved additivity evidence covers classification at 4–8 bits (they
   report additivity breaking <4 bits). Their detector experiments (authoritative info:
   Swin-T/S Mask R-CNN, COCO, 4/5/6-bit) establish detector-level results, but the
   **conditional / repair-from-quantized-base** decision setting is where we measure that the
   context effects are first-order in the decision (Kendall ≈ 0.24–0.42; additive error 5–6 AP
   at K=4); whether an assumed-additive Γ-style policy also mis-allocates under this setting is
   the Gate-B behavioral check.

**However — honest boundary:** "output-aware sensitivity" as a *category* is occupied by BLOB-Q
(via their model-output quadratic). Our novelty cannot rest on "using outputs instead of features".
It must rest on **(N2) conditional / non-additive task utility** + **(N3) its GT-free measurement**
(see `NOVELTY_GATE.md`).

## 4. Verdict

```text
OUTCALsd vs BLOB-Q  : Meaningfully Different (待 strict Γ-reimplementation ranking check in Gate B)
OUTCALsd vs LampQ   : Overlapping — same task/detector; different signal (conditional-measured vs Fisher singleton)
OUTCALsd vs F-LBQ   : Meaningfully Different (and F-LBQ's proxy is empirically mis-ranking)
OUTCALsd vs Reg-PTQ : Different scope (allocation vs calibration)
```
