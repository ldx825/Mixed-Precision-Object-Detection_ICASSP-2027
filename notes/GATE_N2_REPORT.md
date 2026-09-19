# Gate N2 — Hard Novelty Gate (re-issued after Addendum corrections)

Date: 2026-09-12 (Phase II, STEP 4). Basis: corrected prior-work audits (`BLOBQ_METHOD_DERIVATION.md`,
`BLOBQ_vs_CONTEXTUAL_UTILITY.md`, `LAMPQ_METHOD_DERIVATION.md`, `OUTCALSD_vs_PRIOR_EQUIVALENCE_AUDIT.md`,
`OUTCALSD_FORMULATION.md`) + measured evidence (`PHASE2_EXP00_RESULT_AUDIT.md`: S1–S4, additive-error
table, SEL_S1; actionability probe).

## Q1 — Is OUTCALsd mathematically just a simpler empirical version of BLOB-Q's output distortion?

**No — different object, different estimator, different decision variable.**

- BLOB-Q's object is model-output L2 ($\mathbb{E}\|O-\hat O\|^2$) approximated by a second-order
  Taylor quantity whose separability into per-layer terms $\sum_l\Gamma_l(b_l)$ follows from the
  layer-uncorrelatedness Assumption 1 (their Finding 1); allocation = DP over separable costs.
- OUTCALsd's object is the **detector's structured output agreement** (matched-score recovery /
  lost detections / IoU vs the FP teacher) **measured directly on a quantized base state** —
  no curvature model, no additivity assumption, and the decision variable is the **conditional**
  marginal $\hat\Delta(i\mid S)$, not a singleton $\Gamma_l$.
- One is not a truncation of the other: a singleton estimator, however accurate, does not produce
  a context-dependent decision variable. (Boundary kept: "output-aware" as a *category* is
  occupied by BLOB-Q; the novelty rests on conditionality + GT-free measurement, not on the word
  "output".)

**Residual (unverified):** BLOB-Q's detector-side formulation (per authoritative info they report
Mask R-CNN + Swin-T/S on COCO at 4/5/6-bit) was not available to this audit; the conservative
reading "*same framework, detector instantiation unverified*" is used until the primary source is
retrievable. This is why the comparison is behavioral (see Q2/Q4) rather than rhetorical.

## Q2 — Does conditional re-measurement $\hat\Delta(i\mid S)$ create a genuinely different allocation problem?

**Yes — measured, not assumed.**

- Same base, symmetric actions (S3): singleton→conditional ranking agreement only
  Kendall **+0.242** with **25/66** pairwise order reversals (top-3 overlap 2/4→ measured).
- Singleton calib score-recovery → conditional benefit (S4): Kendall **−0.364**, **45/66**
  reversals, top-4 overlap 0/4.
- Additive prediction error for K=4 combos (5k AP): **−4.95 … −6.12 AP** (interaction ≈ 40–50%
  of the singleton sum) — the one-shot composition of singleton marginals is materially wrong
  exactly where the decision is made.
- Full 66-pair interaction matrix + context-rank-reversal analysis (|S|=1,2,3) is **in progress
  (Exp30/31)** and will be reported at Gate M; current evidence already shows the decision
  problem differs.

## Q3 — Does the conditional ranking materially differ from the listed singleton families?

| family | comparison available? | result |
|---|---|---|
| F-LBQ feature ranking (D_feat) | yes (S2; plus Exp03v2 stage/block correlations) | Kendall −0.273 / Spearman −0.441; wrong-signed |
| BLOB-Q-style independent output ranking | yes, via **singleton output stand-in** (calib score-recovery, S4) | Kendall −0.364; top-4 overlap 0/4 |
| LampQ-style independent Fisher ranking | **only via our Fisher-style/singleton stand-ins** (official detection pipeline not public → not reimplemented) | same negative family; wording restricted to "our stand-in behaves poorly in our controlled experiment" |

**Yes for the two directly comparable families; the third only through stand-ins** (stated
explicitly — no claim about official LampQ).

## Q4 — Does the difference improve actual same-budget AP?

- **Controlled mechanism experiment (current):** at equal repair budget (4/12 blocks, 5k box AP):
  singleton-decision stand-in SEL_S1 = **31.39** (≈ chance-level 31.07), FEAT = 22.76,
  conditional OUT = **34.42–34.95**, oracle = 35.32 → conditional decisions beat independent
  rankings by **≈ 2.2–3.6 AP at identical budget**.
- **Caveat (mandated):** this is a controlled mechanism experiment, NOT the final MPQ claim.
  True same-budget MPQ (6.0 / 5.0 / 4.5) head-to-head at matched budgets = Gate B (STEP 8–10).
  The +12.2 AP controlled number vs BASE4 must never be presented as the final improvement.

## Verdict

```text
[Gate N2]

Question:        Is the novelty (context-dependent precision utility) distinct from BLOB-Q / LampQ / Reg-PTQ and actionable?
Exact evidence:  corrected formula-level audits (different object/estimator/decision variable);
                 measured ranking disagreement (S1–S4), additive error 5–6 AP @K=4;
                 controlled same-budget decision gap 31.39 vs 34.95.
Numbers:         Kendall +0.24 (same-base) / −0.27 … −0.36 (singleton families); reversals 25–45/66;
                 conditional-vs-standin decision gap ≈ +3.6 AP (5k, mechanism experiment).
Comparison to strongest prior: behavioral comparison at true budget = Gate B (pending).
What claim is now allowed?    "Detector precision utility is context-dependent and non-additive;
                               singleton rankings mis-allocate in our controlled setting; 64-image
                               GT-free conditional measurement tracks oracle closely."
What claim is still forbidden? "BLOB-Q has no detection experiment" (false); "BLOB-Q assumes additivity"
                               without the exact formula + operational-consequence framing;
                               "LampQ is approximately random" (only stand-in wording allowed);
                               presenting +12.2 AP as the final improvement.
Decision:        CONDITIONAL GO — proceed to STEP 5/6 (66-pair interactions + rank reversal, running),
                 then Gate M. Final novelty confirmation requires Gate M + Gate B.
Next experiment: Exp30/Exp30b (interaction matrix, full triples/quads) → Exp31 rank reversal → Gate M report.
```

## Mandatory risk register (updated)

1. BLOB-Q detector-side formulation not audited from primary source → behavioral (not rhetorical)
   comparison at Gate B; monitor ECVA/author page for camera-ready.
2. LampQ official Fisher pipeline not reimplemented → all LampQ-adjacent statements use
   "Fisher-style stand-in" wording.
3. W-only vs W+A contract deviation — must be stated explicitly in any cross-paper comparison.
4. Cross-paper numbers not comparable (FP 46.0/41.6 v2.x-era vs our 42.65 v3.3 on the same
   checkpoint in mmdet 3.3).
