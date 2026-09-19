# BLOB-Q Method Derivation (formula-level audit)

Source: ECCV 2026 chapter (Springer, DOI 10.1007/978-3-032-37464-6_21), public supplementary
(`papers/related/BLOBQ_supplementary.pdf`, text in `notes/BLOBQ_supp_text.txt`), author page.
Main text is paywalled; all statements below are from the **public** abstract + **full supplementary**.

## 0. One-line positioning

BLOB-Q solves the **global MPQ problem analytically**: approximate the model-level L2 distortion
Γ(O, Ô) via second-order Taylor + a **layer-uncorrelatedness assumption**, show the objective is
**additive** (Γ ≈ Σ_i Γ_i), then solve the decomposed problem with **DP in linear time** while
claiming global optimality (under the assumption).

## 1. Distortion definition

For layer-wise weight perturbation ΔW_l = W_l − Ŵ_l (and similarly activations ΔA_l):

- Second-order expansion of the model output (first-order term ≈ 0 on converged models):

  Ô ≈ O + ½ Σ_l ΔW_l^T H_l^w ΔW_l        (H = Hessian of output w.r.t. layer params)

- Expected L2 distortion expands into **pairwise cross terms**:

  Γ(O,Ô) = E‖O−Ô‖² ≈ Σ_{i,j} E[ (½ΔW_i^T H_i ΔW_i)^T (½ΔW_j^T H_j ΔW_j) ]

- **Assumption 1 (Layer-uncorrelatedness)**: perturbations are (1) zero-mean and
  (2) uncorrelated across layers → all i≠j terms vanish →

  **Γ(O,Ô) ≈ Σ_i E‖½ΔW_i^T H_i^w ΔW_i‖² + Σ_i E‖½ΔA_i^T H_i^a ΔA_i‖²   (Finding 1: Second-Order Additivity)**

- Per-layer term is the second-order quantity ("quantization distortion" Γ_i), estimated with
  Hessian/Jacobian machinery (per their abstract: "quadratically decomposed").

## 2. Decomposition — what it actually depends on

- Relies on **Finding 1** whose proof consumes **Assumption 1** only.
- Cross terms are eliminated *not* by a structural theorem about networks but by the
  **statistical uncorrelatedness assumption**.
- Empirical support in supp:
  - Fig.1: weight-perturbation histograms symmetric → zero-mean plausible.
  - Fig.2: pairwise covariance matrix of layerwise weight perturbations is diagonal →
    uncorrelatedness plausible.
  - Fig.3/4 (additivity): "**at 4 to 5 bits, most of points lie in the diagonal**...
    **when the bit width is less than 4 bits, outlier points start to increase**".
- The additivity evidence retrieved above is at the classification/perturbation level. Per the
  team's **authoritative external information** (2026-09-12 correction), BLOB-Q *does* report
  downstream **object-detection/segmentation experiments** — **Mask R-CNN with Swin-T/S on COCO,
  in 4/5/6-bit settings**. Our retrieved MOESM1 supplementary copy does not contain those
  sections (retrieval limitation of this audit — the Springer page lists only one supplementary
  file; treat the external information as authoritative; the detection details themselves were
  not re-verified against the primary source here).
- No **"repair-from-quantized-base" (conditional)** scenario is documented in anything we
  retrieved or in the public abstract.

## 3. Solver

- After additivity, global problem = per-layer cost/bit curves + one budget constraint →
  **DP over layers × cumulative model size**, linear-time; the paper claims global optimality
  *given the assumption*.
- Supp §2: "Parallel DP-solver Design".

## 4. Detector extension (what is actually public)

| item | status |
|---|---|
| detection/segmentation experiments | ✅ **present** — Mask R-CNN + Swin-T/S, COCO, 4/5/6-bit settings (per authoritative external information, 2026-09-12) |
| in our retrieved MOESM1 supp copy | not located (0 detection-term occurrences; Springer page exposes only this one supplementary file) — **retrieval limitation of this audit**, superseded by the authoritative information above |
| detection in public abstract | not mentioned (abstract claims ImageNet accuracy, ViT-S…ViT-L) |
| calibration count / candidate bits | not stated in public material (standard PTQ calibration; bits 4–8 range) |
| quantized modules | ViT/Swin linear layers; detector-side allocation granularity not audited from primary source |

## 5. Implications for our novelty (why this is the key prior to beat)

1. BLOB-Q's Γ is an **analytic second-order approximation**, whose additivity (their Finding 1)
   is **assumed statistically**; the additivity validations we retrieved are at the
   classification/perturbation level. Because BLOB-Q *does* report detector experiments
   (Mask R-CNN+Swin-T/S, COCO, 4/5/6-bit — authoritative info), any comparison must now be
   **behavioral, not rhetorical**: does an assumed-additive Γ-style allocation strategy agree
   with the *conditional* utility ranking we measure on a detector, and does the decision
   variable (context-dependent marginal utility) change outcomes at equal budget? Our
   singleton-metric stand-in family already shows ≈3.6 AP below measure-what-you-decide at
   equal budget (5k); to be repeated at true 6/5-bit budgets (Gate B).
2. **They themselves report additivity breaking <4 bits** (retrieved supp evidence) — adjacent
   to our aggressive regime; at 4–6 bits they claim it "holds well" for their classification
   settings.
3. Our core observed phenomenon — **context-dependent precision utility / rank reversal under a
   quantized base** (e.g., 6-bit swap gaps 0.24–0.43 pt with OUT 2/2 correct, FEAT 0/2) — is a
   **task-utility-level (AP) claim**, one level above Γ-additivity. **A key Gate-N/B experiment is
   therefore: does BLOB-Q-style additive Γ-ranking agree with measured conditional utility ranking,
   and which one wins at true same budget on a detector?**
4. Differences on paper (to be verified experimentally, not just rhetorically):
   (a) analytic 2nd-order vs direct measurement;
   (b) classification L2 vs detector structured outputs;
   (c) global separable DP vs conditional marginal utility (interaction-aware);
   (d) **BLOB-Q targets 4–8 bits "without hurting accuracy"; we target the aggressive 4–5.5 bit
       detector regime where their own assumption weakens.**

## 6. Residual unknowns (to track)

- Main-text detection experiments (can't verify; monitor author page / ECVA for the camera-ready).
- Their exact Hessian estimator (K-FAC/empirical Fisher/GN) — affects fairness of any
  reimplementation comparison (plan §46: label as reimplementation, align quantizer/calib/budget).
