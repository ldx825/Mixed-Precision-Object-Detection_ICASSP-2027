# Context-Dependent Precision Utility for Mixed-Precision Object Detection

Code release for the paper **"Context-Dependent Precision Utility for Mixed-Precision Object Detection"** (manuscript under review).

- Paper PDF: [`paper/main.pdf`](paper/main.pdf)
- Authors: Dongxu Liu, Linhao Wang, Chenrui Zhu, Dongyang Liu

This repository contains the code that implements the method and produces every number, table and figure in the paper. It is a research release: COCO images, annotations and detector checkpoints are **not** distributed — see [Data](#3-data).

---

## 1. Code overview

### 1.1 What the paper shows and does

**Measurement (Sec. 3).** The dominant mixed-precision recipe scores each layer independently and allocates bits from those singleton scores. On Mask R-CNN + Swin-T, this code measures the detector's *precision utility* in context and shows the independence assumption breaks: pairwise interactions are `0.79×` the scale of singleton effects, `22.6–32.0%` of block rankings reverse across five calibration seeds, and the decision-level cost at an equal budget reaches `3.6` AP.

**Method (Sec. 4, Algorithm 1).** A labels-free allocation procedure that needs only `n = 64` unlabeled images:

1. **Utility** — `Util(b)` runs the test pipeline on the calibration images and matches teacher/student detections per image and class (greedy by score at fixed thresholds), returning `−score_drop` versus the FP teacher.
2. **Phase 1 — conditional greedy** — repeatedly apply the single-level block upgrade that maximises `[Util(b+e_i) − Util(b)] / n_i`, i.e. marginal utility *re-measured in context*, not a singleton score.
3. **Phase 2 — ε-tolerant split-half refinement** — split the calibration set into halves `A/B`, accept the best move with `d_A + d_B > 0` and `min(d_A,d_B) ≥ −ε`; moves are single-level swaps and budget-neutral pair exchanges (`i:+1`, `j:−1`).
4. **Phase 3 — selection** — return the best start by full-64 utility.

**Quantization contract.** Weight-only symmetric min-max uniform fake quantization, per tensor, over the linear groups of the Swin blocks (`attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2` × 3 stages = 12 blocks); biases stay FP. The budget is the parameter-weighted average bit-width over those 12 blocks.

### 1.2 Where the method lives

| Module | Role |
|---|---|
| [`src/quant/fake_quant.py`](src/quant/fake_quant.py) | The quantization contract: `quantize_weight_tensor()` (symmetric min-max, per-tensor), `swin_block_groups()` / `get_block_and_stage_groups()` (the 12-block grid), `GroupQuantApplier` with bit-exact restore. |
| [`src/quant/sanity.py`](src/quant/sanity.py) | Exp02 quantization sanity tests: 16-bit ≈ FP, monotone degradation for 8/4-bit, bit-exact restore after applying/undoing an allocation. |
| [`src/gate_b/exp40_conditional.py`](src/gate_b/exp40_conditional.py) | Phase 1: one conditional-greedy trajectory from the uniform-4 base; prefix truncations give the allocations for lower budgets (hard budget respected at every step). Also captures the first-step `4→8` "static-OUT" signal. |
| [`src/gate_b/exp40b_conditional_full.py`](src/gate_b/exp40b_conditional_full.py), [`exp56_full_greedy_5060.py`](src/gate_b/exp56_full_greedy_5060.py) | Full-budget greedy for fixed targets (4.5/5.5/5.0/6.0), fixing the overshoot of the prefix-truncation snapshots. |
| [`src/gate_b/exp52_split_half.py`](src/gate_b/exp52_split_half.py), [`exp53_sh_greedy.py`](src/gate_b/exp53_sh_greedy.py) | Strict split-half validation: a move/upgrade must improve **both** halves of the 64-image set. |
| [`src/gate_b/exp55_v3_combined.py`](src/gate_b/exp55_v3_combined.py) | The paper's ε-tolerant acceptance rule (`dA + dB > 0` and `min(dA,dB) ≥ −EPS`, `EPS = 0.001`) on combined single-level + pair moves. |
| [`src/gate_b/exp54_pair_swap.py`](src/gate_b/exp54_pair_swap.py) | Budget-neutral pair exchanges (`i:+1`, `j:−1`), which reach reallocations that single-block moves cannot. |
| [`src/gate_b/exp42_refine.py`](src/gate_b/exp42_refine.py), [`exp50_power_refine.py`](src/gate_b/exp50_power_refine.py), [`exp51_final_sprint.py`](src/gate_b/exp51_final_sprint.py) | Refinement variants: deployable swap refinement, 256-image re-validation, multi-start 320-image ensemble verification. |
| [`src/gate_b/measure_signals.py`](src/gate_b/measure_signals.py), [`allocate_static.py`](src/gate_b/allocate_static.py) | Singleton signal measurement and the static baselines: `uniform`, `F-LBQ-reimpl` (feature-MSE curves), `BLOBQ-reimpl` (output-damage curves, greedy + exact additive DP), `LampQ-style` (perturbation-sensitivity scalar), `static-OUT`. |
| [`src/gate_b/exp41_oracle_approx.py`](src/gate_b/exp41_oracle_approx.py) | Oracle-approximation reference (1-swap local search from the conditional allocation). |
| [`src/gate_b/eval_allocations.py`](src/gate_b/eval_allocations.py), [`collect_final_results.py`](src/gate_b/collect_final_results.py) | Full COCO `val2017` (5k) evaluation of every unique allocation and the merged final table. |
| [`src/gate_b/exp56c_budget_completion.py`](src/gate_b/exp56c_budget_completion.py), [`exp57_3bit_ext_40.py`](src/gate_b/exp57_3bit_ext_40.py), [`exp59_3bit_ext_455055.py`](src/gate_b/exp59_3bit_ext_455055.py), [`exp62_3bit_multistart.py`](src/gate_b/exp62_3bit_multistart.py), [`exp66_40_rescue.py`](src/gate_b/exp66_40_rescue.py) | Budget completion and `{3..8}` level-set extensions for the 4.0/4.5/5.0/5.5 budgets. |
| [`src/gate_b/exp60_swins_baseline.py`](src/gate_b/exp60_swins_baseline.py), [`exp61_swins_sweep.py`](src/gate_b/exp61_swins_sweep.py), [`exp63_swins_refine.py`](src/gate_b/exp63_swins_refine.py), [`exp64_baseline_3bit_fairness.py`](src/gate_b/exp64_baseline_3bit_fairness.py), [`exp65_baseline_3bit_4055.py`](src/gate_b/exp65_baseline_3bit_4055.py) | Second-backbone (Swin-S) transfer and the matched-contract baseline comparisons. |
| [`src/diagnostics/exp30_pair_interaction.py`](src/diagnostics/exp30_pair_interaction.py), [`exp30b_full_subsets.py`](src/diagnostics/exp30b_full_subsets.py) | Measured utility tables: 66 pairs, 220 triples, 495 quadruples (`U(S) = −score_drop`), on 64 unlabeled images. |
| [`src/diagnostics/exp31_context_rank_reversal.py`](src/diagnostics/exp31_context_rank_reversal.py), [`exp33_decision_k4.py`](src/diagnostics/exp33_decision_k4.py), [`exp35_seed_pair_validation.py`](src/diagnostics/exp35_seed_pair_validation.py) | Ranking reversals across context sizes / seeds, decision-level impact at `K = 4`, cross-seed validation. |
| [`src/diagnostics/rank_singleton_vs_conditional.py`](src/diagnostics/rank_singleton_vs_conditional.py), [`additive_selection_test.py`](src/diagnostics/additive_selection_test.py) | Can any independent (singleton) score predict the conditional ranking? What would an additive-prior method actually select? |
| [`src/diagnostics/exp34_seed_curve.py`](src/diagnostics/exp34_seed_curve.py), [`exp58_t3_reversal_model.py`](src/gate_b/exp58_t3_reversal_model.py) | Empirical support for the sample-complexity analysis: calibration-size/seed curves for `n ∈ {8..256}`, and the interaction-conditioned reversal-floor model (T3). |
| [`src/diagnostics/actionability_probe.py`](src/diagnostics/actionability_probe.py), [`sweep_v2.py`](src/diagnostics/sweep_v2.py), [`sweep_groupbits.py`](src/diagnostics/sweep_groupbits.py), [`eval5k_points.py`](src/diagnostics/eval5k_points.py) | GT-free proxy probes, the Exp03 block-level perturbation sweeps, and 5k re-evaluation of decisive points. |
| [`src/baseline/exp01_repro_fp.py`](src/baseline/exp01_repro_fp.py), [`exp00_machine_audit.py`](src/baseline/exp00_machine_audit.py), [`make_splits.py`](src/baseline/make_splits.py), [`make_calib_seeds.py`](src/baseline/make_calib_seeds.py) | FP baseline reproduction gate, machine/asset audit, and construction of the calibration/probe splits. |
| [`src/utils/coco_eval_subset.py`](src/utils/coco_eval_subset.py), [`fix_catids.py`](src/utils/fix_catids.py) | COCO evaluation restricted to a probe subset, and category-id repair for saved predictions. |

### 1.3 Paper ↔ code correspondence

| Paper | Code |
|---|---|
| Sec. 3, Table 1 / Fig. 2 — pairwise interaction at `0.79×` singleton scale | `src/diagnostics/exp30_pair_interaction.py`, `exp30b_full_subsets.py` |
| Sec. 3 / Fig. 3 — singleton-vs-conditional ranking under growing context (`22.6–32.0%` reversals over 5 seeds) | `exp31_context_rank_reversal.py`, `exp35_seed_pair_validation.py`, `exp35_analyze_only.py` |
| Sec. 3 — decision-level impact (`3.6` AP at equal budget) | `exp33_decision_k4.py`, `headroom_choice.py` |
| Sec. 4, Algorithm 1, step `Util(b)` | `src/quant/fake_quant.py` + `src/diagnostics/confirm_5k.py::run_calib_dets` |
| Sec. 4, Algorithm 1, Phase 1 (conditional greedy) | `src/gate_b/exp40_conditional.py` (and `exp40b`, `exp56` for fixed targets) |
| Sec. 4, Algorithm 1, Phase 2 (ε-tolerant split-half, Eq. `eps`) | `exp55_v3_combined.py`; strict variants `exp52_split_half.py`, `exp53_sh_greedy.py`; pair moves `exp54_pair_swap.py` |
| Sec. 4, Algorithm 1, Phase 3 (multi-start selection) | `exp50_power_refine.py`, `exp51_final_sprint.py` |
| Sec. 4 — baselines (`uniform`, F-LBQ/BLOB-Q/LampQ re-implementations, oracle-approx) | `allocate_static.py`, `measure_signals.py`, `exp41_oracle_approx.py`, `exp64`/`exp65` (fair `{3..8}` level set) |
| Sec. 5, (T1)/(T2) — Hoeffding sample-complexity bounds, why 64 images suffice | `exp34_seed_curve.py` (calibration-size/seed curves), `exp35_analyze_only.py` |
| Sec. 5, (T3) — interaction-conditioned reversal floor | `src/gate_b/exp58_t3_reversal_model.py` |
| Sec. 6, Table 2 — COCO 5k AP at matched parameter-weighted budgets | `eval_allocations.py`, `run_eval5k.sh`, `collect_final_results.py` |
| Sec. 6 — Swin-S transfer (`+12.5` AP at 4.5-bit) | `exp60_swins_baseline.py`, `exp61_swins_sweep.py`, `exp63_swins_refine.py` |
| Sec. 6 — low-budget (3-bit) level-set study | `exp57`, `exp59`, `exp62`, `exp66` |

### 1.4 Repository layout

```
src/
  quant/          quantization contract (fake quant, block groups) + sanity tests
  baseline/       FP reproduction gate, machine audit, split construction
  diagnostics/    context-dependence measurements, proxies, sweeps, seed curves
  gate_b/         the allocation method (greedy / split-half / pair refinement), baselines, 5k evaluation
  utils/          COCO subset evaluation helpers
scripts/          shell drivers for the experiment chains (see below)
assets/           locked dependency list
data/splits/      calibration (64 / 256-image pools) and probe (512) image-id lists
paper/            compiled paper PDF
notes/, figures/, tables/, outputs/   (not tracked) research notes and experiment artefacts
```

---

## 2. Environment

Reported runs used Python 3.12, `torch 2.3.1+cu121`, `mmcv 2.2.0` (needs the cp312 prebuilt wheel), `mmdet 3.3.0` and `mmengine 0.10.7`. The pinned list is [`assets/requirements_locked.txt`](assets/requirements_locked.txt).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r assets/requirements_locked.txt
```

Detector checkpoints (Mask R-CNN + Swin-T / Swin-S) and the COCO `val2017` images + `instances_val2017.json` are not included; place them where `src/baseline/exp01_repro_fp.py` expects them.

## 3. Data

| Item | Source | Used for |
|---|---|---|
| `val2017` images + annotations | COCO 2017 | FP baseline gate, probe512, and the 5k evaluation of every reported AP |
| `data/splits/calib64_seed0.txt` | 64 images sampled from COCO `train2017` (seed 0) | the labels-free calibration used by the method |
| `data/splits/calib256_seed{0..4}.txt` | five nested 256-image pools | calibration-size / seed robustness |
| `data/splits/probe512_seed0.txt` | 512 `val2017` images | fast intermediate evaluation |

Ground truth is only used to compute the final AP numbers, never inside the allocation loop.

## 4. Running the experiments

> **Paths:** every driver currently resolves the project root from an absolute constant, e.g. `ROOT = Path("/root/autodl-tmp/EviZO-VP/F-LBQ")` in `src/**/*.py` and `ROOT=/root/autodl-tmp/EviZO-VP/F-LBQ` in `scripts/*.sh`, and calls the interpreter as `.venv/bin/python`. Point them at your checkout once before running:
>
> ```bash
> grep -rl "/root/autodl-tmp/EviZO-VP/F-LBQ" src scripts \
>   | xargs sed -i 's#/root/autodl-tmp/EviZO-VP/F-LBQ#/path/to/this/repo#g'
> ```

Gates first — do not continue if either fails:

```bash
# Exp01: FP32 Swin-T Mask R-CNN reproduction (STOP if > 1.0 AP off the model-zoo numbers)
bash scripts/run_exp01.sh
# Exp02: quantization sanity (16-bit ~ FP, monotone 8/4-bit degradation, bit-exact restore)
bash scripts/run_exp02.sh
```

The remaining stages are grouped into drivers. Because several drivers wait for each other by polling `pgrep`, they are meant to be launched in the order of this table (originally a single-machine overnight queue):

| Driver | What it runs |
|---|---|
| `scripts/run_exp03_sweep.sh` | Exp03 block-level perturbation sweeps |
| `scripts/run_analysis.sh` | Exp03 summary (`eval_sweep.py`) + feature-distortion vs AP correlations |
| `scripts/run_gateb_chain.sh` | waits for `measure_signals` → exp40 conditional greedy → exp35 seed validation → static allocations → 5k evaluation |
| `scripts/run_exp3134_chain.sh` | waits for exp30b → exp31 rank reversal → exp33 decision impact → exp34 seed curve |
| `scripts/run_night_finalize.sh` | waits for exp40b → exp42 refine + evaluate at 4.5 / 5.5 → regenerate the verdict |
| `scripts/run_xl_chain.sh` | exp52 split-half validated refinement @ 4.5 / 5.5 / 5.0 |
| `scripts/run_shgreedy_chain.sh` | waits for the xl chain → exp53 split-half greedy |
| `scripts/run_pair_chain.sh` | waits for shgreedy → exp54 budget-neutral pair swaps @ 4.5 / 5.5 / 5.0 / 6.0 |
| `scripts/run_v3_chain.sh` | waits for the pair chain → exp55 ε-tolerant combined refinement |
| `scripts/run_pair_ext_chain.sh` | waits for the v3 chain → exp54 extension |
| `scripts/run_power_chain.sh` | exp50 multi-start power refinement @ 5.5 / 5.0 |
| `scripts/run_final_sprint.sh` | exp51 final sprint (multi-start + 320-image ensemble verification) |
| `scripts/run_attack2_chain.sh` | exp56 full-budget greedy @ 5.0 / 6.0 → exp56c budget completion → exp57 4.0 with 3-bit levels |
| `scripts/run_attack3_chain.sh` | exp59 level-set extension @ 4.5 / 5.0 / 5.5 → exp60 Swin-S baseline |
| `scripts/run_attack4_chain.sh` | waits for exp61 → exp62 multi-start @ 5.0 / 5.5 → exp63 Swin-S refine |
| `scripts/run_attack5_chain.sh` | waits for the attack4 chain → exp64 baseline fairness with `{3..8}` levels |
| `scripts/run_final_collect.sh` | waits for the pair-ext chain → `collect_final_results.py` → final AP table |

Each stage can also be run directly:

```bash
python src/gate_b/exp40_conditional.py      # Phase 1 conditional greedy
python src/gate_b/exp55_v3_combined.py      # Phase 2 epsilon-tolerant refinement
python src/gate_b/eval_allocations.py       # 5k evaluation of the candidate allocations
python src/gate_b/collect_final_results.py  # -> final AP table
```

Artefacts are written under `outputs/` (`outputs/GateB/`, `outputs/Exp30_pair_interaction/`, `outputs/Exp34_seed_curve/`, ...) and `logs/`; both are untracked. Intermediate AP numbers are produced on `probe512`; only the reported table numbers come from the full 5k evaluation.

## 5. Notes and limitations

* The code is exactly the research pipeline that produced the paper. Several chains intentionally overwrite `outputs/*_log.txt` and assume a single GPU (`CUDA_VISIBLE_DEVICES` selects it).
* Calibration utilities are matched against an FP teacher without ground truth; small differences in matching thresholds or in the mmdet/mmcv versions can move the last decimal of a calibration utility.
* `data/splits/` is tracked so that every reported calibration pool is reproducible bit-for-bit; regenerate pools with `src/baseline/make_calib_seeds.py`.
* `src/utils/fix_catids.py` documents a category-id pitfall of the first Exp01 run (`mmdet_label + 1` is not a valid COCO id); regenerated results embed the fix.

## 6. Citation

```bibtex
@misc{liu2026precision,
  title  = {Context-Dependent Precision Utility for Mixed-Precision Object Detection},
  author = {Liu, Dongxu and Wang, Linhao and Zhu, Chenrui and Liu, Dongyang},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## 7. Contact

Dongxu Liu — liudx9924@mails.jlu.edu.cn
