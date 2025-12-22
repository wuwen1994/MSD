# Mask-Free Shadow Detection (MSD)

This repository contains the official implementation of **Mask-Free Shadow Detection (MSD)**, a framework that reformulates shadow detection as an **illumination-induced anomaly localization problem**, eliminating the need for pixel-level shadow mask annotations.

> **Paper**: Mask-Free Shadow Detection  
> **Authors**: Xian-Tao Wu, Wen Wu*, Jifeng Zhu  
> **Affiliations**: Hangzhou Dianzi University, City University of Hong Kong  
> **Code**: https://github.com/wuwen1994/MSD  

---

## 🔥 Highlights

- 🚫 **Mask-free supervision**: No shadow masks are used during training  
- 🌞 **Illumination-aware feature disentanglement (IFD)**  
- 📊 **Dirichlet Process Mixture Model (DPMM)** for shadow-free illumination modeling  
- 🔁 **Self-training refinement** for pixel-accurate shadow masks  
- 🏆 Competitive performance against fully & weakly supervised SOTA methods  

---

## 🧠 Method Overview

MSD formulates shadow detection as **anomaly localization under illumination variation**. Given only **shadow-free images**, the model learns a normal illumination distribution, where shadows emerge as deviations at test time.

### Overall Pipeline

![Framework Overview](figures/fig1.jpg)

**Pipeline stages**:
1. **Illumination-aware Feature Disentanglement (IFD)**  
2. **Shadow-Free Distribution Modeling with DPMM**  
3. **Shadow Anomaly Scoring & Pseudo-label Refinement**

---
## 📂 Dataset

### Shadow Anomaly Detection (SAD) Dataset

We curate a large-scale **shadow-free dataset** from:
- DESOBA
- ISTD
- SRD
- USR
- Internet-crawled natural images

**Key properties**:
- ~4,000 shadow-free images
- Diverse scenes & illumination
- Explicit inclusion of dark materials (non-shadow)

> Note: The dataset is used **only for learning normal illumination**, without any shadow masks.

---

## 🧪 Experiments

### Benchmarks

- **SBU** (Relabeled)
- **ISTD**
- **UCF**

### Metric

- **Balanced Error Rate (BER)**

### Quantitative Results (BER ↓)

| Method | Supervision | SBU | UCF | ISTD |
|------|------------|-----|-----|------|
| MSD (Ours) | Mask-free | **4.37** | **6.14** | **1.11** |
| SSD | Weakly | 5.27 | 6.94 | 1.48 |
| SILT | Fully | 4.19 | 7.23 | 1.16 |

MSD achieves **state-of-the-art or competitive performance**, despite using **zero shadow annotations**.



