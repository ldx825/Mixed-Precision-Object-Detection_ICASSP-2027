# Reg-PTQ Method Derivation (formula-level audit)

Source: CVPR 2024 openaccess (`papers/related/RegPTQ_CVPR2024.pdf`, text `notes/RegPTQ_text.txt`).

## 1. Positioning

**Not a bit-allocation method.** Reg-PTQ is a **full-quantization PTQ scheme** (uniform INT4,
"7.6× computation / 5.4× storage reduction") specialized for **regression-friendly quantization of
detectors**. Core claims: "first to explore regression-friendly quantization", revealing *why*
regressors are hard to quantize.

## 2. Method components

1. **Filtered Global Loss Integration Calibration** — integrates the **global (detection) loss**
   into calibration with a **two-step filtering** mechanism that suppresses the adverse impact of
   **false-positive bounding boxes** (i.e., calibration is performed with detector outputs in the
   loop — teacher/global-loss style).
2. **Learnable Logarithmic-Affine Quantizer** — targets the non-uniform parameter distributions of
   regression structures (bbox deltas etc.).

## 3. Relation to our line of work

- **Overlap:** shared high-level motivation that *local reconstruction losses are not aligned with
  detector quality* — they operationalize this by putting the **global loss** into calibration.
  (Keyword scan: "bit allocation" 0×, "allocation" 0×, "mixed precision" 1×.)
- **Difference:** they **calibrate quantizer parameters under a fixed uniform bit-width**; we
  **allocate bits across blocks** under a fixed budget. Their loop uses global loss with
  (pseudo-)supervision; our utility uses **no GT** and only teacher-vs-quantized detection outputs.
- **Implication for novelty:** cannot claim "first to notice task-vs-reconstruction mismatch in
  detection quantization"; must frame our novelty at the **allocation / conditional-utility /
  non-additivity** level. Reg-PTQ is a **compliant baseline component** (their calibration could
  in principle be combined with any allocation).

## 4. Open items

- Whether their "global loss" uses GT labels/pseudo boxes (needed for a fair "uses-GT" column in the
  equivalence table). Text scan: "pseudo box" 0×; "false positive" appears (filtering); the
  global-loss calibration implies detector outputs in the loop — treat as **uses supervision
  (GT) in calibration** unless contradicted.
