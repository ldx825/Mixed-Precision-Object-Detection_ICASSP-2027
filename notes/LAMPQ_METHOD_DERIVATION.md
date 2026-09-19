# LampQ Method Derivation (formula-level audit)

Source: AAAI 2026 paper (OJS 37489, `papers/related/LampQ_AAAI2026.pdf`, text `notes/LampQ_text.txt`),
code repo (`third_party/LampQ`, based on RepQ-ViT + AdaLog). arXiv id 2511.10004 mentioned in repo.

## 1. Positioning

"Layer-wise Mixed Precision Quantization for Vision Transformers". Explicitly identifies three
limitations of prior ViT MPQ: **(1) coarse granularity, (2) metric-scale mismatch across component
types, (3) quantization-unaware bit allocation**. Claims SOTA on ImageNet (cls), **MS COCO
(detection)**, zero-shot quantization.

## 2. Sensitivity metric (their "type-aware Fisher")

- A **Fisher-based metric**, made "type-aware" (per module type: qkv / proj / fc1 / fc2 …) to fix
  the scale-mismatch problem.
- Sensitivity is **singleton / independent per layer** (one scalar per candidate module) — Fisher
  is a gradient/Hessian-statistics quantity (∂L/∂θ), classically computed over a data batch with a
  supervised objective (needs labels or pseudo-labels; exact recipe not in the public repo).
- Ablation: "type-aware Fisher-based metric (I2) having the strongest impact of 19.96%p" (3-bit DeiT-S).

## 3. Allocation solver

- **Integer Linear Programming** over per-layer bit choices + budget, plus **iterative bit update**
  ("incorporates the quantization feedback by iterative bit updates", search_round=3).
- Repo facts: `calib_size=32`, `optim_size=1024`, `search_round=3`, `steps=6`, `eq_n=128`, FPCS
  scale search; **ILP outcomes are loaded externally** (`utils/others.py: load_bits_from_ilp`);
  Fisher/ILP implementations are **not in the public repo** (classification-only config shipped).

## 4. Detection evidence (public)

Table 4, MS COCO 2014, **Mask R-CNN + Swin-T (also Swin-S / Cascade)**, W/A = **4MP/4MP**:

| method (Swin-T MRCNN) | AP_box | AP_mask |
|---|---|---|
| Full-Precision 32/32 | 46.0 | 41.6 |
| RepQ-ViT 4/4 | 36.1 | 36.0 |
| OPTQ 4/4 | 36.3 | 36.3 |
| ERQ 4/4 | 36.8 | 36.6 |
| AdaLog 4/4 | 39.1 | 37.7 |
| VT-PTQ† 4MP/4MP | 39.2 | 37.7 |
| **LampQ 4MP/4MP** | **39.8** | **38.4** |

- LampQ−best-prior = **+0.6 box / +0.7 mask pt** at 4MP/4MP.
- Note: FP reference here (46.0/41.6) is the *old v2.x-era* number; our same-ckpt reproduction on
  mmdet 3.3 gives 42.65/39.28 — **cross-paper numbers are not comparable** (§46 rule).
- Their detection uses **W+A 4-bit mixed precision**; our current contract is **weights-only** —
  alignment decision required before head-to-head.

## 5. Contrast with OUTCALsd (why it matters for our window)

| aspect | LampQ | OUTCALsd |
|---|---|---|
| sensitivity object | singleton Fisher score per layer | **conditional marginal utility** Δ(g \| S) measured on the *quantized* base |
| feedback | iterative re-ranking with feedback (indirect) | **direct one-shot measurement per candidate** (64 imgs) |
| solver | ILP on additive per-layer costs | greedy conditional selection (S start = quantized base) |
| GT use | likely calibration labels/pseudo-labels (unverified) | **no GT at all** (only FP-teacher detections) |
| failure mode we probe | independent singleton metric can mis-rank under context — measured with **our Fisher-style / singleton stand-in metrics** (official LampQ Fisher pipeline not reimplemented) | immune by construction (measures the context) |

> **Stand-in disclaimer**: all controlled comparisons in this project rank candidate blocks with
> empirical singleton metrics we implemented (singleton damage; singleton calib score-recovery).
> These are *Fisher-style / singleton stand-ins*, **not** the official LampQ type-aware Fisher
> metric (its detection-side implementation is not public). Statements such as "singleton
> rankings underperform" refer strictly to our stand-in family in our controlled setting, not to
> the published LampQ method itself.

## 6. Residual unknowns

- Exact Fisher objective for detection (loss form; uses GT? pseudo boxes?) — extended manuscript
  referenced but not public; monitor arXiv 2511.10004.
- Bit allocation for the Table-4 setting (which layers got which bits; calib count for detection).
