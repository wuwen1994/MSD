# Mask-Free Shadow Detection (MSD)

This repository hosts the code and project page of **Mask-Free Shadow Detection (MSD)**, a framework that reformulates shadow detection as an **illumination-induced anomaly localization** problem. Here, *mask-free* describes the optimization targets of MSD—no pixel-level shadow mask is used as a training target—rather than an absence of supervision in data construction (see the [Dataset](#shadow-anomaly-detection-sad-dataset) section for each source's supervision chain).

> **Paper**: Mask-free shadow detection (*Neural Networks*, under review)
> **Code**: https://github.com/wuwen1994/MSD

---

## 👥 Authors and Affiliations

| Author | Affiliation |
|---|---|
| Wen Wu, Yanzhong Sun, Yuchang Mo | School of Computer Science and Technology, Zhejiang University of Water Resources and Electric Power, Hangzhou, China |
| Lei Yan, Yunze Zhu, Chao Zhang | Qiantang River Basin Center of Zhejiang Province, Hangzhou, China |
| Xian-Tao Wu \* | School of Computer Science and Technology, Hangzhou Dianzi University, Hangzhou, China |

\* Corresponding author: Xian-Tao Wu (<xiantao.hdu.cs@gmail.com>).

---

## 🔥 Highlights

- 🚫 **Mask-free supervision**: No ground-truth pixel-level shadow masks are used as training targets
- 🌞 **Illumination-Aware Feature Disentanglement (IFD)**: Maps patch embeddings into structure-aware and illumination-aware subspaces
- 📊 **Dirichlet Process Mixture Model (DPMM)**: Adaptive prototype modeling of shadow-free illumination patterns
- 🔁 **Iterative pseudo-label refinement**: DenseCRF + self-training with warm-up for pixel-accurate shadow masks
- 🏆 Competitive with state-of-the-art fully and weakly supervised shadow detectors across four benchmarks

---

## 🧠 Method Overview

MSD formulates shadow detection as **anomaly localization under illumination variation**. It learns a normal-illumination distribution from curated **shadow-free images**, generates coarse anomaly maps as initial pseudo-masks, and refines them via DenseCRF and iterative self-training on unlabeled target-domain training images.

**Pipeline stages (Algorithm 1 in the paper)**:
1. **Illumination-Aware Feature Disentanglement (IFD)**: Fine-tune the last two layers of DINOv3-S+/16 with two MLP projectors ($\lambda_1 = 0.5$, $\lambda_2 = 0.3$, $E_{\mathrm{ifd}} = 20$ epochs)
2. **Shadow-Free Distribution Modeling**: Fit a diagonal-covariance truncated DPMM ($C = 80$, $\rho_t = 0.2$, pruning threshold $t_\pi = 10^{-6}$) to shadow-free illumination embeddings
3. **Shadow Anomaly Scoring**: Score target-domain patches by cosine distance to the nearest active prototype, followed by 1st–99th percentile rescaling to $[0, 1]$
4. **Pseudo-Label Refinement**: Upsample, apply DenseCRF, threshold at $0.5$, and self-train an EfficientNet-B3 detector ($E_{\mathrm{shadow}} = 30$ epochs, $E_{\mathrm{warm}} = 10$ warm-up epochs) with class-balanced cross-entropy loss

---

## 📦 Released Code

The code in this repository covers the distribution-modeling and scoring core of the pipeline (**Stages 2–3**): fitting a truncated Dirichlet process mixture to shadow-free embeddings, scoring patch embeddings of target images against the retained components, and evaluating the resulting anomaly maps.

```
.
├── main.py                         # entry point: fit the DPMM, then evaluate
├── generate_feature_diagnostics.py # feature diagnostics for a saved checkpoint
├── config/
│   ├── Diag.yaml                   # diagonal-covariance DPMM (paper default, C=80)
│   ├── Full.yaml                   # full-covariance variant
│   └── Diag.windows.yaml           # Windows-local copy of Diag.yaml
├── src/
│   ├── backbones.py                # DINOv3 / DINOv2 / ViT / ResNet feature extractors
│   ├── data.py                     # dataset and dataloader construction
│   ├── DirichletProcessMixture/
│   │   ├── dpmm.py                 # truncated DPMM: fitting, posterior, scoring
│   │   └── gmm.py                  # GMM baseline
│   ├── train.py                    # online variational / EM fitting of the DPMM
│   ├── evaluate.py                 # anomaly maps from mixture scores
│   ├── test.py                     # metrics and threshold selection
│   ├── visualization.py            # qualitative and diagnostic figures
│   ├── positional_encoding.py      # optional 2-D positional encoding
│   └── utils.py                    # config, checkpoint and tensor helpers
├── docs/ENVIRONMENT.md             # environment notes
└── requirements.txt
```

> **Backbone**: Aligned with Section 5.1 of the paper, the default configuration uses `dinov3_vits16plus` (**DINOv3-S+/16**, 384-dimensional patch embeddings; at $448 \times 448$ resolution with batch size 10, each mini-batch yields $28 \times 28 \times 10 = 7{,}840$ patch embeddings). `DINOv2`, `ViT`, and `ResNet` backbones are also supported in `src/backbones.py`.

Scoring follows Stage 3 of the paper: each patch is scored by its cosine distance to the nearest retained mixture component whose weight exceeds $t_\pi = 10^{-6}$. The evaluation code additionally computes Euclidean distance, negative log-likelihood, and Mahalanobis-style variants for the scoring-function ablations (Table 3 in the paper).

---

## ⚙️ Installation

```bash
conda create -n shadow python=3.10
conda activate shadow
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

The code was developed and tested with PyTorch 2.3.1 + CUDA 11.8. See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the exact package versions.

---

## 📂 Dataset

### Shadow-ANO data layout

The training/evaluation code reads a single root directory that contains one folder per split and per image type:

```
Shadow-ANO/
├── train/
│   ├── good/                      # shadow-free images (normal illumination)
│   └── Ungood/                    # unlabeled shadow images for pseudo-label refinement
├── valid/
│   ├── good/img/
│   └── Ungood/{img,label}/        # label: binary shadow mask (held-out validation)
└── test/
    ├── good/img/
    └── Ungood/{img,label}/
```

Set `dataset.data_root` in the configuration file to the location of this directory.

### Shadow Anomaly Detection (SAD) dataset

The shadow-free reference pool of MSD consists of approximately **3,500 shadow-free images** curated from five sources:

| Source | Approx. Count | Provenance / Supervision Chain |
|---|---|---|
| **DESOBAv2** | ~2,850 | Shadow-removal outputs inheriting upstream mask-based training |
| **ISTD** | 40 | Paired captures from the official ISTD **training** split only |
| **SRD** | 50 | Paired shadow / shadow-free captures |
| **USR** | 60 | Unpaired shadow-free images |
| **Internet** | ~500 | Manually screened to exclude obvious cast shadows while retaining intrinsically dark materials |

**Key properties**:
- Diverse scenes, materials, and lighting conditions with deliberate retention of intrinsically dark non-shadow surfaces
- Holds no shadow-free image from any benchmark test split; scenes visually similar to benchmark test sets were removed
- Used **only for learning normal illumination**; no ground-truth shadow masks are used as training targets

---

## 🚀 Usage

Fit the DPMM on shadow-free embeddings and evaluate on the test split:

```bash
python main.py --config config/Diag.yaml
```

Results (checkpoints, logs and figures) are written to
`results/<dataset_name>/<architecture_name>/`, and the configuration actually used is copied there as `cfg.yaml`.

Key options in `config/Diag.yaml` (aligned with Section 5.1 of the paper):

| Option | Paper Default | Meaning |
|---|---|---|
| `model_name` | `dinov3_vits16plus` | Feature backbone (`DINOv3-S+/16`, 384-d) |
| `resolution` | `448` | Input resolution ($448 \times 448$, $28 \times 28$ patches) |
| `embedding_batch_size` | `10` | Mini-batch size for DPMM fitting (7,840 patches/batch) |
| `feature_resolution_train` / `eval` | `-1` | Keep native patch grid without resampling |
| `density_model.dpmm.max_num_components` | `80` | Truncation bound $C$ of the DP mixture |
| `density_model.dpmm.update_rate` | `0.2` | Exponential smoothing rate $\rho_t$ |
| `density_model.dpmm.gaussian.covariance_type` | `diag` | Diagonal covariances (`diag`, `full`, or `spherical`) |
| `dataset.data_root` | `data/Shadow-ANO` | Path to the Shadow-ANO directory |
| `wandb_log` | `False` | Enable Weights & Biases logging |

---

## 🧪 Experiments

### Benchmarks and Evaluation Protocol

- **Relabeled SBU**: 4,089 training images; evaluated on the 638 relabeled test images
- **ISTD**: 1,870 image triplets
- **CUHK-Shadow**: 10,500 shadow images (7,350 train / 1,050 val / 2,100 test)
- **UCF**: 245 image pairs used strictly for cross-dataset generalization (the SBU-trained detector and threshold are applied directly without adaptation)

Pseudo-masks and self-training use the training splits only. Hyperparameters and the binarization threshold are chosen to minimize BER on a held-out 10% split of each official training set (keeping paired captures, same-scene images, and capture sequences on the same side), then frozen for single-pass test evaluation. All reported numbers are means over five runs.

### Metric

- **Balanced Error Rate (BER ↓)**, with **SER** (shadow error rate) and **NER** (non-shadow error rate)

### Quantitative Results on SBU, UCF, and ISTD (Table 1, BER ↓)

| Method | Year | Supervision | Relabeled SBU | UCF | ISTD |
|---|---|---|---|---|---|
| **MSD (Ours)** | -- | Mask-free | **4.17** | 6.41 | **0.82** |
| FasterSD | 2026 | Fully | 4.34 | 6.61 | 1.25 |
| ShadowAdapter | 2025 | Fully | 5.14 | **6.35** | 0.86 |
| STNet | 2025 | Fully | -- | 6.36 | 1.15 |
| SSD | 2025 | Weakly | 5.27 | 6.94 | 1.48 |
| SdSAM | 2024 | Weakly | 5.43 | 6.55 | 1.14 |
| SILT | 2023 | Fully | 4.19 | 7.23 | 1.16 |
| SDCM | 2022 | Fully | 5.71 | 6.69 | 1.41 |
| FDRNet | 2021 | Fully | 5.93 | 12.91 | 1.55 |
| MTMT | 2020 | Semi | 7.41 | 7.47 | 1.72 |
| DSDNet | 2019 | Fully | 5.60 | 7.59 | 2.17 |
| DSC | 2018 | Fully | 8.08 | 10.54 | 3.42 |
| BDRAR | 2018 | Fully | 6.49 | 7.81 | 2.69 |

### Label-Free and Cross-Dataset Comparison (Table 5, BER ↓)

| Method | CUHK-Shadow | SBU → ISTD | SBU → CUHK |
|---|---|---|---|
| AnomalyDINO | 17.82 | 4.31 | 19.93 |
| PatchCore | 15.65 | 2.92 | 17.21 |
| ShadowScout | 11.74 | 3.48 | 13.52 |
| **MSD (Ours)** | **9.42** | **2.81** | **12.06** |

*(For SBU → ISTD transfer, the 40 ISTD references are excluded from SAD before fitting the source model.)*

---

## 📖 Citation

```bibtex
@article{wu2026msd,
  title   = {Mask-free shadow detection},
  author  = {Wu, Wen and Yan, Lei and Zhu, Yunze and Zhang, Chao and Wu, Xian-Tao and Sun, Yanzhong and Mo, Yuchang},
  journal = {Neural Networks},
  year    = {2026}
}
```

---

## 📄 License and Acknowledgements

Parts of the training and evaluation pipeline (`main.py`, `src/backbones.py`, `src/data.py`, `src/evaluate.py`, `src/train.py`, `src/test.py`, `src/utils.py`, `src/visualization.py`) are adapted from [AnomalyDINO](https://github.com/dammsi/AnomalyDINO), which is distributed under the Apache License 2.0. Those files are used and redistributed here under the same license; see `LICENSE` and `NOTICE`.
