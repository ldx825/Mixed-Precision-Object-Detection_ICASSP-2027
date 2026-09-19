# Context-Dependent Precision Utility for Mixed-Precision Object Detection

Official code and paper materials for the **ICASSP 2027** submission.

> **Code:** https://github.com/ldx825/Mixed-Precision-Object-Detection_ICASSP-2027 · **Paper PDF:** [`paper/main.pdf`](paper/main.pdf)

**Authors:** Dongxu Liu, Linhao Wang, Chenrui Zhu, Dongyang Liu (Jilin University / UESTC)

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | ICASSP 2027 LaTeX source (`main.tex` + `sections/`), figures, compiled `main.pdf` |
| `scripts/` | Experiment drivers and reproduction chains (see below) |
| `src/` | Method implementation: conditional marginal-utility measurement, ε-tolerant split-half search, budget-neutral pairwise refinement, allocation |
| `assets/` | Locked dependency list (`requirements_locked.txt`) |
| `data/splits/` | Calibration / probe split lists used in the paper |
| `figures/`, `tables/` | Paper figures and final result tables |
| `STATUS.md`, `DECISIONS.md` | Experiment status, gates, and decision log |
| `notes/` | Research notes |

Datasets: COCO val2017 (5k) + annotations and detector checkpoints are **not included**.

## Quick start

Environment: `assets/requirements_locked.txt` — torch 2.3.1+cu121, mmcv 2.2.0, mmdet 3.3.0, mmengine 0.10.7 (mmcv requires the cp312 prebuilt wheel).

```bash
# 1) Exp01: FP baseline gate (STOP if > 1.0 AP deviation from the official metafile)
bash scripts/run_exp01.sh
# 2) Exp02: quantization sanity (16/8/4-bit degradation + bit-exact restore)
bash scripts/run_exp02.sh

# 3) Main method chains
bash scripts/run_gateb_chain.sh      # conditional greedy -> seed validation -> static allocation -> 5k eval
bash scripts/run_shgreedy_chain.sh   # shuffle-greedy baseline
bash scripts/run_pair_chain.sh       # budget-neutral pairwise swaps (4.5/5.5/5.0/6.0-bit)
bash scripts/run_final_collect.sh    # -> tables/FINAL_TABLE.md
```

## Compiling the paper

```bash
cd paper && latexmk -pdf main.tex
```

ICASSP template files are included (`spconf.sty`, `IEEEbib.bst`).

## Citation

```bibtex
@inproceedings{liu2027precision,
  title     = {Context-Dependent Precision Utility for Mixed-Precision Object Detection},
  author    = {Liu, Dongxu and Wang, Linhao and Zhu, Chenrui and Liu, Dongyang},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

## Contact

Dongxu Liu — liudx9924@mails.jlu.edu.cn
