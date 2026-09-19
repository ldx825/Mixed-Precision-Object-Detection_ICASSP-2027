# Phenomenon Gate — F-LBQ proxy reliability audit (Exp03 v1 + v2)

Date: 2026-09-12 (~T+7h of topic)
Author: agent (data: outputs/Exp03, outputs/Exp03v2, tables/Exp03v2_all_summary.json)

**Verdict: GO-candidate — feature-alteration proxy shows a systematic mismatch with
detection damage. Proceed to diagnosis phase (plan §26–33). Caveats: small n per
design, probe512 evaluation noise; 5k re-evaluation of key points is the first
post-GO task.**

## Setup

| item | value |
|---|---|
| Detector | mmdet 3.3.0 Mask R-CNN + Swin-T, official 1x ckpt; FP baseline box 42.65 / mask 39.28 (val5k), probe512 bbox AP 43.02 |
| Quantizer | symmetric min-max uniform fake-quant, weight-only, per-tensor, identical across all arms |
| D_feat | relative MSE on calib64 (64 train2017 imgs), backbone stage outputs (C2–C5) pooled; denominator = higher-precision side |
| Baseline probe | probe512_seed0 (from val2017), bbox COCOeval |

## Designs and raw findings

### v1 — single-linear groups (48 groups × {6,4}) — NEGATIVE (unit too fine)
All runs: |ΔAP| < 0.01 AP (bbox + S/M/L) even at 4-bit with rel_mse up to 0.032.
Single linear module ≈ 0.2% of backbone params — below detection threshold.
(22 runs completed; kept as "granularity floor" evidence.)

### v2-A — leave-up (12 blocks; base = ALL blocks 4-bit, AP 17.52)
Repair one block (→8bit), measure AP recovery.
- Sanity: all-blocks-8bit control = 42.70 ≈ FP 43.02 (pipeline verified)
- **Spearman(D_feat, AP_up) = −0.441 (p=0.15, Kendall −0.27) — WRONG SIGN**
- Only 1/12 blocks (BLK2.3) yields positive repair benefit; repairing most
  blocks makes AP *worse* (cross-layer interaction; non-additive).

### v2-B — stage-level independent perturbation (8 runs)
- **Spearman(D_feat, damage) = +0.595 (p=0.12, Kendall +0.55) — right sign, weak**
- Rank inversion: STG0 D_feat=0.096 (largest) but damage=1.1 AP (smallest);
  STG2 D_feat=0.039 (3rd) but damage=10.7 AP (largest). 2.5× feat gap vs 10× damage gap.

### v2-C — block-only perturbation from FP (12 runs × 4bit)
- **Spearman(D_feat, damage) = −0.161 (p=0.62) — none**
- Damage ceiling low (max 1.5 AP), dominated by noise.

## Cross-design pattern (qualitative, robust)

```text
shallow blocks (STG0/1, BLK0.x/1.x):  D_feat LARGE  → task damage SMALL
deep blocks   (STG2/3, BLK2.3/3.x):   D_feat SMALL  → task damage LARGE
```

F-LBQ's feature-alteration proxy systematically ranks "large feature change" as
more important, but large feature change concentrates in early layers while
task sensitivity concentrates in deep layers (STG2 = 6 blocks, STG3) — the two
orderings disagree.

**This directly contradicts F-LBQ Fig.1's interpretation** ("intermediate layers
contribute the least to detection accuracy"): in our measurements the
intermediate/deep stages dominate detection damage (STG2 b4 damage 10.7 AP;
STG2 repair BLK2.3 +6.5 AP), while early stages with the largest feature
alteration are nearly task-inert.

## Gate decision (plan §25)

STOP condition = Spearman ≥ 0.8 AND Kendall ≥ 0.7 AND no obvious matched-MSE
inversions. **Not met** (observed −0.44 / +0.60 / −0.16 across designs, with
explicit rank inversions). Therefore:

- NOT "proxy is good → STOP" ⇒ the follow-up topic is **alive**
- Phenomenon direction: proxy mismatch exists ⇒ **GO-candidate**

### Caveats (must be resolved before strong claims)
1. n = 8–12 per design; p ≈ 0.1–0.15 individually (direction evidence, not significance)
2. probe512 evaluation noise ~±0.3–0.5 AP; block-only design underpowered
3. leave-up semantics mix "block importance" with interaction effects; the
   negative-sign correlation needs re-checking at 5k resolution
4. D_feat denominator convention in v2 = quantized/repaired side (documented;
   differs slightly from v1/Exp03 FP-denominator)

## Next tasks (diagnosis phase, plan §26–§33)

1. **5k re-evaluation** of decisive points: leave-up {BLK2.3, BLK3.0, BLK0.0,
   BLK0.1}, stage {STG0, STG2} × {b4}, block-only {BLK3.1, BLK0.0} (≈10 runs
   × 6 min)
2. Scale decomposition (AP_S/M/L) on those points
3. Error-type decomposition (cls/box/mask), fg/bg decomposition
4. Legal-proxy search (teacher outputs, fg-weighted features, confidence
   weighting) → Actionability Gate (plan §34)
