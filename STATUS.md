# STATUS — F-LBQ Follow-up

Updated: 2026-09-12, elapsed ~1.5h (T+0 = session start of F-LBQ topic)

## Phase: first-round asset audit & environment setup (plan §71)

### Done
- Project dirs created under `/root/autodl-tmp/EviZO-VP/F-LBQ` (only allowed root)
- Machine scan: no COCO/MMDet/Swin ckpt/F-LBQ resources pre-existing (logs/initial_asset_scan.txt)
- COCO val2017 (5k ✓) + annotations (instances_val2017.json ✓) downloaded & verified
- F-LBQ supplementary obtained (SigPort; PDF title = F-LBQ supp; `papers/FLBQ/sigport_supp.pdf`, text in `notes/FLBQ_supp_text.txt`)
- F-LBQ main paper: not public (IEEE paywall, no arXiv, no author copy) → documented
- Official code: none found (repo hunt closed ~15 min < 60 min cap) → `F-LBQ-compatible reconstruction` route
- BLOB-Q (author's ECCV 2026 follow-up, code "coming soon") discovered → novelty flag, monitor
- `assets/ASSET_MANIFEST.md` written; `notes/FLBQ_IMPLEMENTATION_AUDIT.md` written
- venv installed: torch 2.3.1+cu121 / torchvision 0.18.1 / mmcv 2.2.0 (cp312) / mmdet 3.3.0 / mmengine 0.10.7 / pyarrow 25.0.1 (numpy 2.5.3 present; mmcv.ops imports OK; mmdet mmcv-version check patched to <2.3.0)
- Exp00 done (machine_audit.json); requirements_locked.txt (53 pkgs)
- calib64 (64 imgs from 2 HF train2017 parquet shards, seed0) + probe512_seed0 created
- **Exp01 DONE: FP baseline box 42.65 / mask 39.28 on val2017 5k — matches official mmdet v3.3.0 metafile (42.7/39.3) within 0.05 AP → GATE PASS** (category_id fix applied; see D008)
- **Exp02 DONE: all sanity checks PASS** — 16bit≈FP (ΔAP<0.5pt), 8bit mild (±0.2 AP noise), 4bit degrades 44.9→20.0 AP (no crash), restore bit-exact (0.0 diff). See outputs/Exp02/sanity_summary.json

### Phase II status (2026-09-12)

**Last Experiment:** Phase II Exp00 audit + Gate N analyses (S1–S4, additive error, SEL_S1)
**Current Gate:** N — **CONDITIONAL GO** → proceed to STEP 8 (freeze contract) & STEP 9 (true 6/5-bit)

**Key Numbers (artifact-verified, see notes/PHASE2_EXP00_RESULT_AUDIT.md):**
- FP 42.65 (5k) / 43.02 (probe512); BASE4 5k 17.67
- K=4 combos (5k): FEAT 22.76 / **SEL_S1 (indep-ranking) 31.39** / OUT512 34.42 / **OUTCALsd 34.95** / ORAC 35.32; oracle-gain recovery 97.9%
- 6-bit: U6 42.13; swap gaps 0.24–0.43 pt (OUT 2/2, FEAT 0/2)
- Singleton→conditional Kendall: +0.42 / −0.27 / **+0.24** / −0.36 (S1–S4); additive error K=4: −4.9…−6.1 AP

**Strongest Prior Comparison:** BLOB-Q (no public detection evidence; additivity theorem; main-text paywall risk) · LampQ (Mask R-CNN+Swin-T 4MP/4MP 39.8/38.4, +0.6 over prior; Fisher singleton + ILP) — same-budget head-to-head NOT yet run

**Current Novelty Statement:** Detector precision utility is context-dependent/non-additive; singleton rankings mis-allocate (~random-level, 3.6 AP below conditional at same budget); a GT-free 64-image conditional measurement recovers 97.9% of the oracle improvement.

**Evidence Against Novelty:** "output-aware" category occupied by BLOB-Q (classification/analytic); SEL_S1 is an empirical stand-in (not BLOB-Q's exact Γ); W+A vs W-only alignment unresolved.

**Next Cheapest Discriminating Experiment:** STEP 8 freeze QUANTIZATION_CONTRACT (incl. W+A decision) → Exp26 true 6-bit head-to-head (Uniform / F-LBQ-reimpl / additive-Γ reimpl / LampQ-style / OUTCALsd).

**Explicit Stop Condition:** If true same-budget Ours ≤ strongest prior +0.2 AP at all of {6, 5.5, 5} bit without size/time advantage → STOP_REPORT_PHASE2.

### Risks
- Disk 35G free; calib via HF streaming (no full train2017 zip, 19G not feasible)
- No nvcc → must use prebuilt mmcv cp312 cu121/torch2.3 wheel (verified exists)
- BLOB-Q (ECCV 2026) may cover part of the global-distortion idea → check before any method claims (plan §38 Novelty Gate)
