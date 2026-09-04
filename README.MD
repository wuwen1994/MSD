# Mask-Free Shadow Detection (MSD)

This repository is the project page for **Mask-Free Shadow Detection (MSD)**, a framework that reformulates shadow detection as an **illumination-induced anomaly localization problem**. Here, *mask-free* means that no manually annotated, task-specific pixel-level shadow mask is used as a training target; MSD still uses curated shadow-free images and automatically generated pseudo masks.

> **Paper**: Mask-Free Shadow Detection  
> **Affiliations**: Hangzhou Dianzi University, City University of Hong Kong  
> **Code**: https://github.com/wuwen1994/MSD  

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

### Overall Pipeline

**Pipeline stages**:
1. **Illumination-aware Feature Disentanglement (IFD)**  
2. **Shadow-Free Distribution Modeling with DPMM**  
3. **Shadow Anomaly Scoring & Pseudo-label Refinement**

---
## 📂 Dataset

### Shadow Anomaly Detection (SAD) Dataset

We curate a large-scale **shadow-free dataset** from:
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

Under the protocol reported in the paper, MSD achieves competitive BER without using manually annotated, task-specific pixel-level shadow masks as training targets. The complete code, evaluation scripts, configuration files, and SAD manifest are planned for release with the paper.






