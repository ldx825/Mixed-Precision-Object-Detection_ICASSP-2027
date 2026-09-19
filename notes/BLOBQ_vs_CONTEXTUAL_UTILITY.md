# BLOB-Q vs Contextual Utility — Formula-by-Formula Audit (Addendum §4/§5)

Date: 2026-09-12. Status: prepared with corrected factual basis.

**Factual basis & corrections**

- BLOB-Q (ECCV 2026, Springer DOI 10.1007/978-3-032-37464-6_21) **does have downstream
  object-detection/segmentation experiments** including Mask R-CNN with Swin-T/S on COCO and
  4/5/6-bit mixed-precision settings — **authoritative external information (2026-09-12
  correction)**. Our retrieved supplementary copy (single MOESM1 file) does not contain those
  sections; that is a **retrieval limitation of this audit**, not evidence about the paper.
- The derivations below that cite "supp" come from the retrieved MOESM1 text
  (`notes/BLOBQ_supp_text.txt`). Items not present there are marked **unverified from primary
  source**.

---

## Q1. What exact distortion does BLOB-Q optimize?

Global **model-output distortion** between full-precision output $O$ and quantized output
$\hat O$:

$$\Gamma(O,\hat O) = \mathbb{E}\|O-\hat O\|^2 .$$

Note this is an **output-space L2** quantity (their retrieved derivation is classification-side;
detector-side instantiation of $\Gamma$ is *unverified from primary source*). It is **not** a
feature-space error and **not** a task metric (AP).

## Q2. What Taylor/Jacobian approximation is used?

**Second-order expansion** of the model output w.r.t. per-layer parameter and activation
perturbations ($\Delta W_l$, $\Delta A_l$), first-order term dropped on converged models:

$$\hat O \approx O + \tfrac12 \sum_l \Delta W_l^{\top} H_{l}^{w}\,\Delta W_l
\;+\; \tfrac12 \sum_l \Delta A_l^{\top} H_{l}^{a}\,\Delta A_l,$$

with $H_l$ the (output) Hessian/Jacobian-derived curvature for layer $l$. Per-layer terms are
then estimated with Hessian/Jacobian machinery ("quadratically decomposed" per their abstract).

## Q3. How are per-layer / per-block terms formed?

Squaring the expansion yields an **expected distortion** that expands into **pairwise cross
terms** across layers:

$$\Gamma \approx \sum_{i,j} \mathbb{E}\Big[\big(\tfrac12 \Delta W_i^{\top} H_i \Delta W_i\big)^{\!\top}
\big(\tfrac12 \Delta W_j^{\top} H_j \Delta W_j\big)\Big] + \text{(activation analog)}.$$

The diagonal ($i=j$) terms are the **per-layer contributions** $\Gamma_l$; the off-diagonal
($i\neq j$) terms are **cross terms**.

## Q4. How are cross terms handled or ignored?

**Not structurally removed — statistically assumed away.**

**Assumption 1 (layer-uncorrelatedness)**: weight/activation perturbations are (a) zero-mean
and (b) **uncorrelated across layers**, which makes all $i\neq j$ expectations vanish. This is
their **Finding 1 (Second-Order Additivity)**:

$$\Gamma(O,\hat O) \;\approx\; \sum_l \mathbb{E}\big\|\tfrac12 \Delta W_l^{\top} H_l^{w} \Delta W_l\big\|^2
\;+\; \sum_l \mathbb{E}\big\|\tfrac12 \Delta A_l^{\top} H_l^{a} \Delta A_l\big\|^2 .$$

Retrieved supp evidence: symmetric perturbation histograms (zero-mean plausible); diagonal
pairwise-covariance matrix (uncorrelatedness plausible); additivity scatters — "at 4 to 5 bits,
most of points lie in the diagonal … when the bit width is less than 4 bits, outlier points
start to increase."

## Q5. What precise mathematical step permits decomposition?

Exactly two moves, in order:

1. **Taylor + Assumption 1** make the *model distortion* equal to a **sum of per-layer
   distortions** $\Gamma \approx \sum_l \Gamma_l$ (Finding 1).
2. Because both the objective and the constraint (total model size) are then **separable across
   layers**, the global allocation problem becomes a **per-layer cost/optimal-bit curve** joined
   only by one budget constraint — the exact precondition for dynamic programming.

> **Safe phrasing to reuse (Addendum §5)**:
> *BLOB-Q approximates global model distortion by the second-order Taylor quantity
> $\mathbb{E}\|O-\hat O\|^2 \approx \sum_l \mathbb{E}\|\frac12\Delta W_l^\top H_l\Delta W_l\|^2
> +\sum_l \mathbb{E}\|\frac12\Delta A_l^\top H_l^a\Delta A_l\|^2$ under the
> layer-uncorrelatedness Assumption 1, under which the objective separates into independent
> per-layer terms, allowing layer-wise optimization with DP.*

## Q6. What does its DP optimize exactly?

After Finding 1: choose a bit width per layer from the candidate set to **minimize the sum of
estimated per-layer distortions $\sum_l \Gamma_l(b_l)$** subject to the **total model-size
budget**:

$$\min_{\{b_l\}} \; \sum_l \Gamma_l(b_l) \quad \text{s.t.} \quad \sum_l \mathrm{size}(b_l) \le C .$$

**DP over (layer index × cumulative size)**; claimed **linear time**. Note the objective is the
*approximated* distortion, and all costs are **singleton quantities** $\Gamma_l(b_l)$ — no term
depends on the bit-width choices of other layers.

## Q7. Under which assumptions is the resulting allocation "global"?

Global optimality is claimed **for the decomposed problem**, conditional on:

1. Assumption 1 (layer-uncorrelatedness) holding for the actual quantization perturbations;
2. per-layer quality of the $\Gamma_l$ estimators;
3. separability of the constraint (size budget);

i.e., optimality is *relative to the additive approximation*. The authors themselves report the
additivity evidence weakening below 4 bits (retrieved supp).

## Q8. Does the detector extension use the exact same decomposition?

**Unverified from primary source.** Per authoritative external information, BLOB-Q reports
Mask R-CNN + Swin-T/S on COCO at 4/5/6-bit, but whether the detector-side pipeline computes
$\Gamma_l$ on detector outputs (e.g. classification head logits / box regression / mask heads),
and whether the same DP + additivity assumption governs allocation there, **has not been
verified by this audit**. Until the primary source is available, the conservative assumption is
*same framework, detector instantiation unverified* — which is exactly why our Gate-B comparison
must be **behavioral** (see below), not rhetorical.

## Q9. Which parts of our OUTCALsd are genuinely different?

| axis | BLOB-Q | OUTCALsd (ours) |
|---|---|---|
| estimated object | model-output L2 (classification-side derivation retrieved) | **detector structured outputs**: matched-score recovery, lost detections, IoU (measured) |
| estimator | analytic 2nd-order + Hessian/Jacobian, per-layer | **direct measurement on a quantized base** (no curvature model) |
| context | singleton $\Gamma_l(b_l)$; other layers' bits do not enter | **conditional**: score of block $i$ measured *given the current repaired set $S$* |
| additivity | **assumed** (Assumption 1 → Finding 1) | **not assumed; falsified by measurement** (pair/quad interaction matrix, rank reversal) |
| optimization | DP on separable costs, claims global optimality *given assumptions* | conditional greedy on measured marginal utility (post-Gate-B) |
| GT | none | none |

**Boundary (kept honest):** "output-aware" as a *category* is occupied by BLOB-Q. Our novelty
does not rest on the word "output" — it rests on **conditional dependence**
$\Delta(i\mid S)$ vs singleton $\Gamma_l$, and its **GT-free measurement** from 64 unlabeled
images.

---

## Operational consequence test (Addendum §5) — the right way to attack

Do **not** attack "BLOB-Q assumes additivity" in the abstract. Test the **operational
consequence** of the decomposition:

> *Can independent block contributions correctly predict which block should receive extra
> precision **after other blocks have already changed precision**?*

Our test (already running):

1. **Exp30**: all 66 pairs $(i,j)$ on calib64, U6 base → interaction matrix
   $I_{ij} = U(\{i,j\}) - U(\{i\}) - U(\{j\})$.
2. **Exp30b/Exp31**: full triples + quadruples → context rank reversal for $|S|=1,2,3$:
   ranking by $\Delta(i\mid S)$ vs singleton ranking — Kendall $\tau$, Spearman, pairwise
   reversal rate, top-1 mismatch, top-$k$ overlap.
3. **Gate-B**: static singleton ranking stand-ins vs conditional OUT at true same budget
   (6.0 / 5.0 / 4.5 first).

If the conditional ranking materially differs (Gate M) **and** same-budget AP prefers the
conditional allocation (Gate B), the operational consequence of the additive decomposition is
empirically violated *for detector precision utility* — a stronger and safer claim than
disputing their assumption in principle.

## What claim would be allowed after positive Gates M+B?

> For detector MPQ in this regime, per-block precision utility is **context-dependent**: the
> contribution of a block changes after other blocks' precision changes, so allocations derived
> from independent singleton contributions (any additive/singleton scoring family, incl. the
> BLOB-Q-style framework as instantiated by our stand-ins) can mis-rank the next repair; a
> GT-free 64-image conditional measurement fixes the allocation decision and improves the
> same-budget AP–bit Pareto frontier.

## What remains forbidden (Addendum §24/§0)

- "BLOB-Q has no detection experiment" — forbidden (it does).
- "BLOB-Q assumes additivity" *without* the exact formula + the operational-consequence framing.
- "LampQ is approximately random" — forbidden; only "our Fisher-style / singleton stand-in
  behaves poorly in our controlled experiment."
- Any claim that OUTCALsd's gain over stand-ins is the final improvement (that is the
  controlled mechanism experiment; Gate B owns the final claim).
