# NOVELTY_AUDIT — corrected index (Addendum §3 compliance)

Date: 2026-09-12. This file is the corrected index for the novelty audit referenced by the
Correction & Execution Addendum (its "`notes/NOVELTY_AUDIT.md`" entry).

## Corrections applied (Addendum §0/§3)

1. **BLOB-Q detection experiments — corrected fact basis.**
   - Forbidden: "BLOB-Q has no detection experiment."
   - Correct: BLOB-Q **has** downstream object-detection/segmentation experiments including
     **Mask R-CNN with Swin-T/S on COCO at 4/5/6-bit** (authoritative external information).
     Our retrieved supplementary copy (single MOESM1 file) does not contain those sections —
     documented as a **retrieval limitation of this audit**, not a fact about the paper.
   - Corrections written in: `BLOBQ_METHOD_DERIVATION.md`, `NOVELTY_GATE.md` (risk register),
     `OUTCALSD_vs_PRIOR_EQUIVALENCE_AUDIT.md` (master table + §3), `GATE_N2_REPORT.md`.

2. **LampQ wording — corrected.**
   - Forbidden: "LampQ is approximately random."
   - Allowed: "our Fisher-style / singleton stand-in behaves poorly in our controlled experiment."
   - Stand-in disclaimer added to `LAMPQ_METHOD_DERIVATION.md` §5; stand-in labeling added to
     `NOVELTY_GATE.md` (SEL_S1 row).

3. **BLOB-Q treatment as strongest conceptual prior** — formula-by-formula audit in
   `BLOBQ_vs_CONTEXTUAL_UTILITY.md` (9 questions + safe phrasing + operational-consequence
   test design).

4. **OUTCALsd mathematical reconstruction** — `OUTCALSD_FORMULATION.md` upgraded:
   explicit Û_OUT(S), Δ̂(i|S), matching rule (per-image/class, REF score≥0.3, greedy IoU≥0.5),
   pooled averaging, perturbation mechanism, GT-free statement.

5. **Novelty target frozen** (Addendum §1): *precision utility in detector MPQ is
   context-dependent* — pillars A (context dependence), B (rank reversal), C (non-additivity),
   D (conditional GT-free measurement). Gate N2 report: `GATE_N2_REPORT.md` (CONDITIONAL GO).

## Canonical files

- `NOVELTY_GATE.md` — Gate N record + risk register (kept as the successor of the original
  novelty audit notes; all "no detection" statements removed).
- `GATE_N2_REPORT.md` — hard Gate N2 with Q1–Q4.
- `OUTCALSD_vs_PRIOR_EQUIVALENCE_AUDIT.md` — master comparison table + decision-level evidence.
- Mechanism evidence: `outputs/Exp30_pair_interaction/`, `outputs/Exp31_context_rank_reversal/`
  (Exp30/30b/31/33; Gate M report pending).
