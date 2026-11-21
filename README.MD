# Mask-Free Shadow Detection (MSD)

This repository hosts the code and the project page of **Mask-Free Shadow Detection (MSD)**, a framework that reformulates shadow detection as an **illumination-induced anomaly localization** problem. Here, *mask-free* means that no manually annotated, task-specific pixel-level shadow mask is used as a training target; MSD still uses curated shadow-free images and automatically generated pseudo masks.

> **Paper**: Mask-Free Shadow Detection (Neural Networks, under review)
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

- 🚫 **Mask-annotation-free supervision**: No manually annotated, task-specific pixel-level shadow masks are used as training targets
- 🌞 **Illumination-aware feature disentanglement (IFD)**
- 📊 **Dirichlet Process Mixture Model (DPMM)** for shadow-free illumination modeling
- 🔁 **Self-training refinement** for pixel-accurate shadow masks
- 🏆 Competitive performance against fully & weakly supervised SOTA methods

---

## 🧠 Method Overview

MSD formulates shadow detection as **anomaly localization under illumination variation**. It learns a normal illumination distribution from curated **shadow-free images**, generates anomaly-derived pseudo masks, and refines those masks by self-training on unlabeled target-domain images.

**Pipeline stages**:
1. **Illumination-aware Feature Disentanglement (IFD)**
2. **Shadow-Free Distribution Modeling with DPMM**
3. **Shadow Anomaly Scoring & Pseudo-label Refinement**

---

## 📦 Released Code

The code in this repository covers the distribution-modeling and scoring part of the pipeline (Stages 2–3): fitting a truncated Dirichlet process mixture to shadow-free illumination embeddings, scoring patch embeddings of target images against the retained components, and evaluating the resulting anomaly maps.

```
.
├── main.py                        # entry point: fit the DPMM, then evaluate
├── generate_feature_diagnostics.py# feature diagnostics for a saved checkpoint
├── config/
│   ├── Diag.yaml                  # diagonal-covariance DPMM (main setting)
│   ├── Full.yaml                  # full-covariance variant
│   └── Diag.windows.yaml          # Windows-local copy of Diag.yaml
├── src/
│   ├── backbones.py               # DINOv2 / ViT / ResNet feature extractors
│   ├── data.py                    # dataset and dataloader construction
│   ├── DirichletProcessMixture/
│   │   ├── dpmm.py                # truncated DPMM: fitting, posterior, scoring
│   │   └── gmm.py                 # GMM baseline
│   ├── train.py                   # EM-style fitting of the DPMM
│   ├── evaluate.py                # anomaly maps from mixture scores
│   ├── test.py                    # metrics and threshold selection
│   ├── visualization.py           # qualitative and diagnostic figures
│   ├── positional_encoding.py     # optional 2-D positional encoding
│   └── utils.py                   # config, checkpoint and tensor helpers
├── docs/ENVIRONMENT.md            # environment notes
└── requirements.txt
```

Scoring follows Stage 3 of the paper: a patch is scored by its (cosine) distance to the nearest retained mixture component whose weight exceeds `1e-6`. The evaluation code additionally reports negative-log-likelihood and Mahalanobis-style variants, which are the baselines compared in the ablation study.

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
│   └── Ungood/                    # shadow images used for pseudo-label refinement
├── valid/
│   ├── good/img/
│   └── Ungood/{img,label}/        # label: binary shadow mask
└── test/
    ├── good/img/
    └── Ungood/{img,label}/
```

Set `dataset.data_root` in the configuration file to the location of this directory.

### Shadow Anomaly Detection (SAD) dataset

The shadow-free images of MSD are curated from:
- DESOBAv2
- ISTD
- SRD
- USR
- Internet-crawled natural images

**Key properties**:
- ~3,500 shadow-free images
- Diverse scenes & illumination
- Explicit inclusion of dark materials (non-shadow)
- The 40 selected ISTD shadow-free images are all from the official training split
- Selected DESOBAv2, ISTD, SRD, and USR images were screened against SBU, UCF, and ISTD for exact and near-duplicate content; no detected cross-dataset duplicate or near-duplicate was retained

> Note: SAD is used **only for learning normal illumination** and does not provide manual pixel-level shadow masks. Manual screening of Internet images is a data-curation cost.

---

## 🚀 Usage

Fit the DPMM on shadow-free embeddings and evaluate on the test split:

```bash
python main.py --config config/Diag.yaml
```

Results (checkpoints, logs and figures) are written to
`results/<dataset_name>/<architecture_name>/`, and the configuration actually used is copied there as `cfg.yaml`.

Key options in `config/Diag.yaml`:

| Option | Meaning |
|---|---|
| `model_name` | feature backbone (`dinov2_vits14`) |
| `resolution` | input resolution (448) |
| `density_model.dpmm.max_num_components` | truncation bound of the DP mixture |
| `density_model.dpmm.gaussian.covariance_type` | `diag`, `full`, or `spherical` |
| `density_model.dpmm.update_rate` / `schedule` | EMA/EXP update of the mixture during fitting |
| `dataset.data_root` | path to the Shadow-ANO directory |
| `wandb_log` | enable Weights & Biases logging |

---

## 🧪 Experiments

### Benchmarks

- **SBU** (Relabeled)
- **ISTD**
- **UCF** (held-out cross-dataset evaluation)

Target-domain pseudo-label refinement is performed only on the SBU and ISTD training sets. UCF images and labels are not used for training, validation, threshold selection, or self-training; the SBU-trained detector is applied directly to UCF.

### Metric

- **Balanced Error Rate (BER)**

### Quantitative Results (BER ↓)

| Method | Supervision | SBU | UCF | ISTD |
|------|------------|-----|-----|------|
| MSD (Ours) | Mask-annotation-free | **4.17** | 6.41 | **0.82** |
| SSD | Weakly | 5.27 | 6.94 | 1.48 |
| SILT | Fully | 4.19 | 7.23 | 1.16 |

---

## 📖 Citation

```bibtex
@article{wu2026msd,
  title   = {Mask-Free Shadow Detection},
  author  = {Wu, Wen and Yan, Lei and Zhu, Yunze and Zhang, Chao and Wu, Xian-Tao and Sun, Yanzhong and Mo, Yuchang},
  journal = {Neural Networks},
  year    = {2026}
}
```

---

## 📄 License and Acknowledgements

Parts of the training and evaluation pipeline (`main.py`, `src/backbones.py`, `src/data.py`, `src/evaluate.py`, `src/train.py`, `src/test.py`, `src/utils.py`, `src/visualization.py`) are adapted from [AnomalyDINO](https://github.com/dammsi/AnomalyDINO), which is distributed under the Apache License 2.0. Those files are used and redistributed here under the same license; see `LICENSE` and `NOTICE`.
