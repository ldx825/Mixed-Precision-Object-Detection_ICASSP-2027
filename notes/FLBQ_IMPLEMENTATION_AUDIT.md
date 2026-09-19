# FLBQ_IMPLEMENTATION_AUDIT — 主论文/官方代码审计

Date: 2026-09-12 (session T+~1h)
Budget used for repo hunt: ~15 min (cap 60 min) — closed, no official code.

## 1. Sources checked

| Source | Result |
|---|---|
| arXiv search "F-LBQ" (API: ti/abs) | ❌ no results |
| Author homepage (xuk114.github.io) publications | ❌ F-LBQ not listed (only LPViT/ICCV23 pruning/ICIP22-23) |
| SigPort supp page | ✅ PDF obtained (`papers/FLBQ/sigport_supp.pdf`) — title inside PDF: "F-LBQ ... Supplementary Materials" (page title misleadingly says F-PTQ) |
| GitHub repo search: "F-LBQ", "F-PTQ" | ❌ no matching official repos |
| Author GitHub (github.com/Akimoto-Cris) repos | ❌ no F-LBQ/F-PTQ quantization repo (LPViT, RD_PRUNE only) |
| IEEE Xplore DOI 10.1109/ICASSP55912.2026.11462477 | paywalled — main text NOT available |
| ICASSP workshop page (cmsworkshops.com PaperNum=15978) | not fetched (no PDF expected; keep DOI as citation) |

**Conclusion**: no official code, no public main text. Follow plan §11: build an
**"F-LBQ-compatible diagnostic reconstruction"** on official MMDetection. Never claim
"reproduction of F-LBQ numbers".

## 2. Verified facts from supplementary (authoritative)

From `notes/FLBQ_supp_text.txt` (2 pages, Feb 6 2025):

1. Calibration: **64 samples randomly selected from training set**.
2. F-LBQ applied to **fine-tuned ViT backbones**; search optimal bit-allocation for
   **transformer layers**, minimizing **alteration on backbone feature** caused by quantization.
3. Mask-RCNN **detection/segmentation heads**: standard uniform quantization + vanilla scaling factors.
4. **Channel-group-wise allocation** used in F-LBQ experiments.
5. F-LBQ-LG (Lagrangian solver): binary search λ on power set 2^R, range [2^-100, 0];
   tolerance ε=0.02 on `|Bit(q(Φ(1:L))) − B| ≤ ε`; asymptotic cost Θ(96L); runs <1s on CPU mostly.
6. Min bit-width constraint: at least 1~2-bit lower than target bit rate (hyperparameter) to
   prevent large layerwise variance.
7. **Fig. 1 (Mask-RCNN-Swin-T)**: DP & LG solvers differ mostly in **earlier layers**; both
   allocate **lowest bit rates to intermediate layers** — authors' claim: "those layers
   contribute the least to the detection accuracy".

**This claim is exactly the audit target**: intermediate layers' low bit allocation rests on
feature-alteration proxy ≈ detection damage ranking.

## 3. Reconstruction plan (Discovery phase)

| Component | Decision | Note |
|---|---|---|
| Detector | mmdet 3.3.0 `mask-rcnn_swin-t-p4-w7_fpn_1x_coco` + official 1x ckpt | box AP≈46.0 / mask AP≈41.6 reference; verify exact zoo number in Exp01 |
| Quantizer | symmetric min-max uniform fake-quant (weights; per-tensor for discovery, channel-group later) | same quantizer across all arms (plan §11) |
| Discovery unit (§18) | Swin **block-level linear groups** (attn.qkv, attn.proj, mlp.fc1, mlp.fc2 per block; 4 groups × 12 blocks ≈ 48 groups) | coarse enough to sweep; refine to channel-group only if GO |
| Perturbation (§19) | Quantize ONE group to b ∈ {6,4}; keep all else FP | Exp03 |
| Feature distortion (§21) | D_feat = mean_i ||F_i^FP − F_i^(l,b)||² / (||F_i^FP||² + ε) on calib64; also raw MSE, cosine, SQNR | F = **backbone stage outputs**; record per-stage (C2..C5) AND pooled last stage; main text definition unknown → documented assumption |
| Probe set (§20) | 512 val2017 images fixed seed0, coverage S/M/L | Stage 1 for all (l,b) |
| Full COCO | only 8–12 discriminating points later (Stage 2) | |
| GT damage (§22) | D_AP = AP_FP − AP_(l,b) via pycocotools, box/mask, overall + S/M/L + AP50/75 | diagnostic only, never in final calibration objective |

## 4. Unknowns (must stay honest in notes)

- Exact feature layer(s) F-LBQ used for the alteration objective; exact quantizer
  (symmetric/asymmetric, granularity, clamping, rounding) and scaling rule; how
  channel-group-wise grouping is formed; which "finetuned ViT backbones" exactly
  (Swin-T for the figure, but main comparisons may cover more). All are reconstructed
  choices — documented, uniform across arms; first-phase conclusions scoped to
  "proxy reliability audit under F-LBQ-compatible setup" (plan §11-4).

## 5. Next actions (plan §71 order)

- [x] ASSET_MANIFEST.md
- [x] main paper (unavailable→documented) + supplementary (obtained)
- [x] official repo check (none; <60min)
- [x] FLBQ_IMPLEMENTATION_AUDIT.md (this file)
- [ ] venv install finish → requirements_locked.txt
- [ ] Exp00 machine audit
- [ ] Exp01 FP Swin-T Mask R-CNN on COCO val2017 5k (gate: >1.0 AP deviation = STOP)
- [ ] Exp02 quant sanity (A–D)
- [ ] calib64 (train2017 subset) + probe512
- [ ] Exp03 single-group perturbation probe
- [ ] Exp04 D_feat vs ΔAP plot + correlations
- [ ] Phenomenon Gate decision (deadline T+8h)
