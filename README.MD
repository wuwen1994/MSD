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

![Framework Overview](figures/fig3.png)

**Pipeline stages**:
1. **Illumination-aware Feature Disentanglement (IFD)**  
2. **Shadow-Free Distribution Modeling with DPMM**  
3. **Shadow Anomaly Scoring & Pseudo-label Refinement**

---

## ✨ Illumination-aware Feature Disentanglement (IFD)

To overcome the illumination invariance bias of foundation models (e.g., DINOv2), we disentangle patch embeddings into:

- **Structure subspace** (illumination-invariant)
- **Illumination subspace** (shadow-sensitive)

![IFD Illustration](figures/fig1.png)

This design enables robust discrimination between **true shadows** and **intrinsically dark objects**.

---

## 📊 Shadow-Free Distribution Modeling

We model the illumination subspace using a **Dirichlet Process Mixture Model (DPMM)**:

- Non-parametric & adaptive
- Compact memory footprint
- Efficient inference compared to kNN-based memory banks

![DPMM Visualization](figures/fig2.png)

Shadows are detected as **low-similarity outliers** under the learned illumination distribution.

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

---

## 🔁 Pseudo-label Refinement

Due to patch-level anomaly maps, we apply:

1. Upsampling + DenseCRF
2. Iterative self-training with a lightweight segmentation network (e.g., U-Net)

This progressively sharpens boundaries and suppresses noise.

---

## 🧩 Applications Beyond Shadows

MSD generalizes naturally to **anomaly segmentation** tasks:

- 🏭 Industrial defect detection  
- 🏥 Medical image anomaly segmentation (e.g., BMAD)

The DPMM-based modeling is especially effective under **limited normal data**.

---

## ⚠️ Limitations

- Requires a collection of **shadow-free images**
- Performance depends on initial anomaly map quality
- Not strictly unsupervised (but fully annotation-free)

---

## 📦 Installation (Coming Soon)

```bash
git clone https://github.com/wuwen1994/MSD.git
cd MSD
pip install -r requirements.txt

