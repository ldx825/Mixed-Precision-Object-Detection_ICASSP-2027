# ASSET MANIFEST — F-LBQ Follow-up (updated: 2026-09-12)

## Machine

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER, 32760 MiB |
| Driver / CUDA | 595.71.05 / CUDA 13.2 (host) |
| Disk | /root/autodl-tmp: 80G total, ~36G free |
| Conda base | /root/miniconda3, Python 3.12.3 (torch 2.8.0+cu128 present, NOT used for this project) |
| System python | /usr/bin/python3.10 |

## Project environment (planned → installed)

| Item | Value |
|---|---|
| venv | `$ROOT/.venv` (Python 3.12, conda base interpreter, NO system-site-packages) |
| torch | 2.3.1+cu121 (installing) |
| torchvision | 0.18.1+cu121 (installing) |
| mmcv | 2.2.0 (cp312 wheel from openmmlab cu121/torch2.3 index) |
| mmdet | 3.3.0 |
| mmengine | 0.10.7 |
| numpy | 1.26.4 |
| pycocotools | pip |
| Status | INSTALLING (log: logs/venv_install.log) |

## Datasets

| Item | Path | Status |
|---|---|---|
| COCO val2017 (5k images) | data/val2017.zip | ✅ downloaded (815 MB, zip OK), unzipping |
| COCO annotations (trainval2017) | data/annotations_trainval2017.zip | ✅ downloaded (253 MB, zip OK), unzipping |
| COCO train2017 | NOT present | ⚠️ 19G full zip not feasible on 36G disk → plan: HF parquet subset for calib64 only (see DECISIONS) |
| Calib split | data/splits/calib64_seed0.txt | ⬜ TODO (64 images, from train2017 subset) |
| Probe split | data/splits/probe512_seed0.txt | ⬜ TODO (512 images from val2017) |

## Models

| Item | Path | Status |
|---|---|---|
| Mask R-CNN + Swin-T (1x, mask_rcnn_swin-t-p4-w7_fpn_1x_coco) | checkpoints/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth | ⏳ downloading (mmdet v2.0 zoo; ~180MB) |
| Config | mmdet 3.3.0 builtin: configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py | via pip install mmdet |
| Reference numbers (from plan/supp) | box AP ≈ 46.0, mask AP ≈ 41.6 (to verify against exact model-zoo number) | |

## Papers

| Item | Path | Status |
|---|---|---|
| F-LBQ supplementary (SigPort, "F-PTQ" title) | papers/FLBQ/sigport_supp.pdf (140 KB, 2 pages) | ✅ downloaded; text: notes/FLBQ_supp_text.txt |
| F-LBQ main paper | — | ❌ not public (IEEE ICASSP 2026 paywall; DOI 10.1109/ICASSP55912.2026.11462477; no arXiv version found) |
| F-LBQ supp key facts | see notes/FLBQ_supp_text.txt | ✅ 64 calib imgs; feature-alteration objective; channel-group-wise; LG solver (ε=0.02, binary search 2^R, [2^-100,0], Θ(96L), <1s CPU); min bit-width target-1~2; Fig.1: Mask-RCNN-Swin-T, lowest bits at intermediate layers ("contribute least to detection accuracy" — TO AUDIT) |
| ICASSP 2026 paper page | https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=15978 | not yet fetched |
| ECCV2024 08994 (K. Xu, LPViT pruning — related author work) | papers/FLBQ/ECCV2024_08994.pdf | ✅ downloaded (not F-LBQ-related quant) |

## Official code

| Item | Status |
|---|---|
| F-LBQ / F-PTQ official repo | SEARCHING (≤60 min budget; GitHub search + author page) |
| Fallback | official MMDetection Swin-T detector + F-LBQ-compatible reconstruction (plan §11) |

## Papers queue (novelty audit later)

- HAWQ, BRECQ, PTQ4ViT, APQ-ViT, RepQ-ViT, Reg-PTQ CVPR2024, etc. → notes/NOVELTY_AUDIT.md (stage 2)
