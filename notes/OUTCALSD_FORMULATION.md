# OUTCALsd — Mathematical Formulation (recovered from code)

Code of record:
`src/diagnostics/confirm_5k.py` (Part A), `src/diagnostics/actionability_probe.py` (`churn`),
`src/diagnostics/headroom_choice.py` / `exp04_budget6.py` (selection + verification).

## 1. Definitions — the estimator (not a code name)

Quantization units ("blocks"): the Swin-T backbone's 12 attention/FFN weight groups
U = {attn.w_msa.qkv, attn.w_msa.proj, ffn.layers.0.0, ffn.layers.1} × 3 stages × 4 blocks.
Everything else (heads, FPN, LN, conv stem) stays FP throughout (contract:
`notes/QUANTIZATION_CONTRACT.md`).

- Model state S (repair set): all 12 groups at base bits b0 ∈ {4, 6} (symmetric min-max,
  per-tensor, weights-only fake-quant), **except** g ∈ S at 8-bit. Activations remain FP
  (contract deviation note).
- **Utility of a state**:  Û_OUT(S) := − score_drop( state(S) ; FP teacher )   (score_drop: §2)
- **Conditional marginal utility**:  Δ̂(i | S) := Û_OUT(S ∪ {i}) − Û_OUT(S)
- Singleton: Δ̂(i | ∅) with all blocks at b0, repairing only i.
- **OUTCALsd (one-shot, K=4)**: the 4 blocks with largest Δ̂(· | ∅); every candidate is
  measured *inside the current base context* (not from the FP model). Sequentially conditional
  measurement (Δ̂(· | S_t), iterative) is the post-Gate-B greedy method (Addendum §15).
- Calibration set: **64 unlabeled images** (calib64 seed0, train2017 subset).
- Teacher references: FP model detections. No labels are loaded; GT is used nowhere.

## 2. score_drop — exact definition (recovered from `churn`, code of record)

Both REF (FP teacher) and Q (quantized candidate) detection sets are produced by the standard
mmdet test pipeline (`inference_detector`, default NMS + `max_per_img=100`), one pass over the
64 calib images (ndarray input, PIL-loaded PNGs; image_id = local index 0..63).

For each image x and each class c:

1. REF side filter: F = {r ∈ REF_x : c(r)=c, score(r) ≥ 0.3}   (fixed at 0.3)
2. Q side: Qc = {q ∈ Q_x : c(q)=c}   (no score filter on the Q side)
3. Greedy one-to-one matching, REF-driven: iterate r ∈ F by decreasing score; match r to the
   unmatched q ∈ Qc with maximal IoU if IoU ≥ 0.5 (else r stays unmatched).
4. score_drop = **unweighted mean** of (score(r) − score(q)) over all matched pairs, pooled
   across all images (no per-image / per-class weighting; no extra confidence weighting).
5. Secondary signals (same pooled convention): lost_frac = 1 − #matched / #F_total;
   new_frac = 1 − #matched / #Q_total; iou_drop = 1 − mean IoU over matched pairs.

**Boxes only** (bbox + class + score). Masks, proposals, and logits are not compared.

## 3. Selection rule

    Ŝ = top-K blocks by Δ̂(g | ∅)   (K = 4 in reported experiments; one-shot)

- **OUT512 variant**: identical formula, computed on 512 val images instead of calib64
  (diagnostic upper bound; not deployable-pure; both are GT-free).
- The signal is **measured inside the current base context** (every candidate re-parses the base
  state), unlike singleton Fisher/feature scores measured from the FP model.
- Composition to K>1 is one-shot (top-K of singletons-in-base-context), **not** iterative
  re-measurement; iterative Δ̂(· | S_t) comes after Gate B.

## 4. Cost

- 1 (BASE4) + 12 (repairs) = **13 detector forwards × 64 images ≈ 2–3 min GPU** per allocation decision.
- No GT, no retraining, no per-layer Hessian.

## 5. Verified behavior (artifact-backed)

| evidence | result |
|---|---|
| predicts per-config damage (stage/block-only, probe512) | Spearman 0.72–1.00 (vs D_feat −0.16…+0.60) |
| 6-bit base actionability (48 configs) | score_drop 0.870 / iou_drop 0.884 (D_feat 0.573) |
| deployed selection quality (5k) | OUTCALsd 34.95 vs ORAC 35.32 (97.9% of oracle gain over BASE4) |
| 6-bit swaps | OUT 2/2 correct directions (FEAT 0/2) |

## 6. Formulation-level caveats

- Score-matching uses detector outputs only → inherits any bias of the detector's own NMS/max-det cap(100).
- Matching threshold (IoU 0.5 / score 0.3) is a design choice; sensitivity to these constants is an
  ablation item (cheap: recompute from stored detections).
- Δ̂(g | ∅) is a *measurement in the base context* per candidate; the K>1 composition is
  one-shot. The gap between one-shot composition and true sequential conditional selection is
  exactly what the Exp30/Exp31 interaction study quantifies (post-Gate-B conditional greedy
  addresses it directly).
