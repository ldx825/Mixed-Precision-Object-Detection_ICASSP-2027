# QUANTIZATION CONTRACT — Phase II (frozen 2026-09-12)

All Phase II comparisons (Exp26–29 head-to-head) MUST use this contract. Any deviation must be
recorded here with a dated note.

## Model & codebase

| item | value |
|---|---|
| Detector | Mask R-CNN + Swin-T, official mmdet 3.3.0 config `mask-rcnn_swin-t-p4-w7_fpn_1x_coco` |
| Checkpoint | `checkpoints/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth` (official) |
| mmdet / mmcv / torch | 3.3.0 / 2.2.0 / 2.3.1+cu121 (venv `.venv`) |
| FP reference (5k bbox/mask) | 42.65 / 39.28 (matches official metafile 42.7/39.3) |

## Data

| item | value |
|---|---|
| Eval | COCO val2017 (5k), pycocotools bbox + mask, maxDets [1,10,100] |
| Calibration | `calib64_seed0` (64 images, **no labels used anywhere**); seed1/2 for robustness |
| Probe (development) | `probe512_seed0` |

## Quantization units & quantizer

- Allocation units: **12 Swin blocks** (each = {attn.w_msa.qkv, attn.w_msa.proj,
  ffn.layers.0.0, ffn.layers.1} weights). Stage-level analyses where stated.
- **Primary (Gate B phase 1): weights-only, symmetric min-max, per-tensor fake-quant**
  (identical to Phase I; same code `src/quant/fake_quant.py`).
- Candidate bits: {4, 5, 6, 7, 8}; base configurations U4 (all-4) and U6 (all-6) as defined per exp.
- Heads / LayerNorm / embeddings / biases: FP. (Deviations vs prior papers' W+A noted in comparisons.)

## Budget & cost

- **Primary budget**: parameter-weighted average bit = Σ n_i·b_i / Σ n_i over the 12 blocks.
- Secondary: model size (MB) of quantized weights; report both.
- All comparisons at **identical** primary budget.

## W+A alignment decision (recorded)

- Priors' detector results (LampQ Table 4 etc.) use W+A quantization. Our phase-1 head-to-head is
  **W-only**, an internal-consistency design (all methods share the same quantizer).
- **Decision:** run Gate B (Exp26/27) on W-only first; if it passes, immediately add a
  **W+A variant** (activations: symmetric min-max per-tensor, scales from calib64) to align with
  prior settings for the final paper table. All W+A numbers must go through the same contract.

## Reporting (every head-to-head point)

bbox AP / AP50 / AP75 / AP_S / AP_M / AP_L, mask AP / AP50 / AP75, effective avg bit, model size MB,
calibration time (wall clock of the allocation search incl. candidate forwards), peak GPU memory.

## Method labels (no conflation)

- `F-LBQ-official` (not obtainable) vs `F-LBQ-reimpl` (feature-MSE + LG-style search) vs
  `FEAT` (feature-MSE top-k, controlled baseline) — must be labeled distinctly.
- `additive-Γ stand-in` (singleton empirical ranking; SEL_S1) vs `BLOB-Q-reimpl` (if Hessian
  approximation implemented) — label distinctly.
- `LampQ-style` (Fisher singleton + ILP, if implemented) vs `LampQ-official` numbers (only as
  sanity reference, never as the same-budget comparison).

## Change log

- 2026-09-12: initial freeze (W-only primary; W+A extension planned post-Gate-B).
