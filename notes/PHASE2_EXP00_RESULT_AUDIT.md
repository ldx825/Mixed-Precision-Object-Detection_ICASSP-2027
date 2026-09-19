# Phase II Exp00 — Current Result Artifact Audit

Date: 2026-09-12 (Phase II, STEP 1–2)
Rule: artifact > chat memory. All values below re-read from the stored JSON artifacts.

## Headline numbers (recomputed from artifacts)

| Claim | Artifact Path | Recomputed? | Value |
|---|---|---:|---:|
| FP (val2017 5k, bbox AP) | `outputs/Exp01/metrics_fixed.json` | ✅ read | **42.65** |
| FP (5k, mask AP) | same | ✅ | 39.28 |
| FP (probe512, bbox AP) | `outputs/Exp03v2_headroom/headroom_metrics.json` (`ref.FP`) | ✅ | 43.02 |
| BASE4 (5k, box AP) | `outputs/Exp03v2_5k/metrics_5k.json` | ✅ | **17.67** |
| BASE4 (probe512) | headroom_metrics `ref.BASE4` | ✅ | 17.52 |
| FEAT_K4 (5k) | `outputs/Exp03v2_headroom_5k/metrics_5k.json` | ✅ | **22.76** (probe512 22.55) |
| OUT512 (5k) | same | ✅ | **34.42** (probe512 34.45) |
| OUTCALsd (5k) | same + `selection_meta.json` | ✅ | **34.95** |
| ORACLE (5k) | same | ✅ | **35.32** (probe512 35.80) |
| RAND_K4 (probe512 only) | headroom_metrics | ✅ | 31.07 |
| Oracle-gain recovery (OUTCALsd) | computed | ✅ | (34.95−17.67)/(35.32−17.67) = 17.28/17.65 = **97.9%** |
| OUTCALsd − FEAT (5k) | computed | ✅ | **+12.19 AP** |
| U6 uniform-6bit (probe512) | `outputs/Exp04_budget6/budget6_summary.json` | ✅ | **42.13** (−0.89 vs FP) |
| 6-bit swap #1 (BLK3.0\|BLK3.1) | same | ✅ | X 41.03 vs Y 41.46, gap **0.43 pt**, true=Y; pred OUT=Y ✓ / FEAT=X ✗ / ORAC=Y ✓ |
| 6-bit swap #2 (BLK2.3\|BLK2.4) | same | ✅ | X 41.46 vs Y 41.70, gap **0.24 pt**, true=Y; pred OUT=Y ✓ / FEAT=X ✗ / ORAC=X ✗ |
| sel_calib_sd (deployable) | headroom_5k `selection_meta.json` | ✅ | [BLK2.4, BLK2.3, BLK3.1, BLK2.2] (overlap 3/4 with ORAC [2.3,3.0,3.1,2.4]) |

**Deltas vs chat memory: none > 0.05 AP.** All previous claims stand.

## Correct wording (mandated)

> OUTCALsd recovers ≈97.9% of the **Oracle improvement over BASE4** in the controlled repair experiment.

Forbidden: "OUTCALsd reaches 97.9% of Oracle performance"; "ours beats F-LBQ by 12.2 AP" (must be
labeled as controlled setting: all-4bit base + repair 4/12).

## Artifact inventory (key dirs)

```
outputs/Exp01/            FP baseline (5k, bbox+mask, metrics_fixed.json)
outputs/Exp02/            quant sanity A–D
outputs/Exp03/             v1 fine-grained sweep (negative result, 22 runs)
outputs/Exp03v2/          block-level runs (33) incl. BASE4, UP_*, STG*, BLK*
outputs/Exp03v2_5k/       12-point 5k re-eval (metrics_5k.json)
outputs/Exp03v2_headroom/         K=4 selection comparison (probe512)
outputs/Exp03v2_headroom_5k/      4 combos on 5k + selection_meta.json
outputs/Exp04_budget6/    6-bit base: 25 single-block + 2 swap pairs (budget6_summary.json)
tables/actionability_probe.json   legal-signal correlations (probe512)
notes/PHENOMENON_GATE.md, notes/ACTIONABILITY_GATE.md
DECISIONS.md D001–D013
```

## Environment / contract candidates (for §12 QUANTIZATION_CONTRACT freeze)

- mmdet 3.3.0 + torch 2.3.1+cu121, mmcv 2.2.0; official ckpt `mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth`
- FP 42.65 (5k) matches official v3.3 metafile 42.7/39.3
- Quantizer so far: **weights only, symmetric min-max per-tensor fake-quant** (activations FP);
  NOTE: LampQ/full-quantization priors use W+A quantization — contract must decide W-only vs W+A for Phase II head-to-head.
- calib64 seed0 (HF train2017 subset), probe512_seed0
