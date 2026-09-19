# Actionability + Headroom Gate — quick check (preliminary)

Date: 2026-09-12 (~T+7.5h)
Status: **PASS (preliminary)** — legal no-GT signals strongly predict detection damage and
yield large measurable gains in a controlled budget experiment.
Caveats: probe512, single seed, transductive selection; deployable (calib64-based)
selection and 5k confirmation are in progress.

## 1. Actionability probe (plan §31–33)

Legal signals computed from **FP-teacher vs quantized detections** (no GT):
lost_frac (FP dets without match), new_frac, score_drop (matched pairs), iou_drop.
Matching: same image + same class + IoU ≥ 0.5, score ≥ 0.3 for FP side.

Spearman with damage (stage/block-only: AP_FP − AP; leave-up: y = AP_up):

| proxy | stage (n=8) | block-only (n=12) | leave-up (n=12) |
|---|---|---|---|
| D_feat (F-LBQ) | +0.595 | −0.161 | −0.441 |
| lost_frac | **+0.976** | **+0.607** | **−0.944** |
| score_drop | **+0.976** | **+0.734** | **−0.874** |
| iou_drop | **+1.000** | **+0.720** | −0.706 |
| new_frac | +0.048 | −0.189 | +0.350 |
| (leave-up vs BASE4 churn, lost_frac) | — | — | +0.893 (p<0.001) |

**Gate §34 STOP condition** (all legal proxies Spearman < 0.4) is clearly NOT met —
best signals reach 0.72–1.00 with p < 0.05, far above the 0.6 "clearly stronger than
baseline feature-MSE" bar.

## 2. Headroom choice experiment (plan §35–36 quick version)

Budget: repair exactly K=4 of 12 blocks (4bit→8bit) on top of all-4bit base.
probe512 bbox AP (FP 43.02; BASE4 17.52):

| strategy | selection | AP | Δ vs FEAT |
|---|---|---|---|
| FEAT (F-LBQ-style, top D_feat) | BLK0.0, 1.0, 0.1, 1.1 | 22.55 | — |
| RAND | seeded random 4 | 31.07 | +8.5 |
| **OUT (legal signal)** | BLK3.1, 2.3, 3.0, 2.5 | **34.45** | **+11.9** |
| ORAC (oracle benefit order) | BLK2.3, 3.0, 3.1, 2.4 | 35.80 | +13.2 |

- Legal-signal allocation approaches oracle within **1.35 AP**
- F-LBQ-style selection performs **worse than random** (−8.5 AP), confirming the
  directional flaw of the feature-alteration proxy (invests budget into shallow
  blocks with large feature change but near-zero task value)
- Headroom threshold in plan §36 (oracle ≥ +0.5 box AP over proxy) is exceeded by
  >20× in this setting

## 3. Full-5k confirmation + deployable selection (added T+~9h)

EXTREME setting (all-4bit base, repair K=4), **val2017 5k box AP**:

| combo | selection | 5k AP | vs BASE4 |
|---|---|---|---|
| BASE4 | — | 0.1767 | — |
| FEAT | D_feat top-4 | 0.2276 | +5.1 |
| OUT512 | probe512 legal top-4 | 0.3442 | +16.8 |
| **OUTCALsd** | **calib64 legal top-4 (deployable, 64 unlabeled imgs)** | **0.3495** | **+17.3** |
| ORAC | true top-4 | 0.3532 | +17.7 |
| FP | — | 0.4265 | — |

- Deployable selection reaches **97.9% of the oracle gain**, only 0.37 pt below ORAC
- **+12.2 pt over F-LBQ-style selection** at full 5k resolution
- Signal: deployable choice used calib64 score-drop (largest score recovery upon
  repair); selection [BLK2.2, BLK2.3, BLK2.4, BLK3.1] overlaps oracle 3/4

## 3b. Realistic budget (uniform-6bit base)

- U6 = 42.13 probe512 (only **−0.89 pt** vs FP 43.02 → optimization ceiling ~0.9 pt)
- Equal-size swap, constant budget: direction gap **0.24–0.43 pt/pair**;
  legal OUT predicts the better direction **2/2**, FEAT 0/2, ORAC (additive) 1/2
- 6-bit actionability (48 single-block configs): D_feat 0.573 vs
  lost_frac 0.764 / score_drop 0.870 / iou_drop 0.884 (all p<0.001)

## 4. Required follow-ups before strong claims

1. **Deployable selection**: recompute OUT selection using calib64 (64 unlabeled
   train images) instead of probe512 outputs; check selection stability
2. **5k confirmation** of OUT/FEAT/ORAC combos
3. **Realistic budget**: average ~5–6 bit (not extreme all-4bit base); per-parameter
   cost weighting instead of block-count budget
4. **Head-to-head vs F-LBQ full pipeline** (channel-group-wise + LG solver)
5. Multi-seed robustness (calib seeds 0/1/2)
