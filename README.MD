# MeT: A Graph Transformer for Semantic Segmentation of 3D Meshes

**Unofficial PyTorch Implementation**

## 📄 Paper
**Title:** *MeT: A Graph Transformer for Semantic Segmentation of 3D Meshes*  
**Author:** Wen Wu (Unofficial Version)

> Original Paper Reference:  
> Vecchio, G., Prezzavento, L., Pino, C., Rundo, F., Palazzo, S., & Spampinato, C.  
> *Computer Vision and Image Understanding*, 2023.

---

## ✅ Environment Setup

### Recommended Versions
```txt
Python 3.10
CUDA 11.8
torch==2.3.1
tensorboard==2.20.0
scikit-learn==1.7.2
scipy==1.15.3
pillow==12.0.0
protobuf==6.33.0
pymesh==1.0.2
pymeshlab==2025.7
numpy==1.26.4
PyYAML==6.0.3
rtree==1.4.1
matplotlib==3.8.0
trimesh==4.9.0
```

---

## 📂 Dataset

### Download Links
| Dataset | Link | Code | Target Directory |
|--------|------|------|----------------|
| Human  | https://pan.baidu.com/s/1Xv_-Tybn2xclz8dsZ4F3pA | `jtjx` | `data/raw/Human` |
| Coseg  | https://pan.baidu.com/s/1Y22-Z7JscMim0LGqJNPSQA | `nim7` | `data/raw/coseg` |

### Directory Structure
```
data/
 └── raw/
     ├── Human/
     └── coseg/
          ├── aliens/
          ├── chairs/
          └── vases/
```

---

## 🛠️ Data Preprocessing

Includes: **feature extraction**, **clustering**, **padding**

```bash
cd data

python preprocess.py --input_root raw/coseg/aliens --output_root processed/coseg/aliens --lam 8 --lap_k 16 --f_pad 1500
python preprocess.py --input_root raw/coseg/chairs --output_root processed/coseg/chairs --lam 8 --lap_k 16 --f_pad 1000
python preprocess.py --input_root raw/coseg/vases --output_root processed/coseg/vases --lam 8 --lap_k 16 --f_pad 1000
python preprocess.py --input_root raw/Human --output_root processed/Human --lam 8 --lap_k 16 --f_pad 1500
```

Processed data will be saved in:
```
data/processed/
```

---

## 🚀 Training

```bash
python train.py --config config/vases.yaml
```

You may modify YAML configs for different datasets.

---

## 🔍 Testing

```bash
python test.py --config config/vases.yaml
```

Results will be saved under the configured experiment directory.

---

## 🎨 Visualization

```bash
python vis.py --obj_path a.obj --seg_path a.seg --save_dir out --no_show
```

- `obj_path`: Input mesh  
- `seg_path`: Predicted segmentation  
- `save_dir`: Output folder  
- `--no_show`: Disable GUI display (optional)

---

## 📚 Citation

If you use this repository for academic purposes, please cite the original paper:

```bibtex
@article{vecchio2023met,
  title={MeT: A graph transformer for semantic segmentation of 3D meshes},
  author={Vecchio, Giuseppe and Prezzavento, Luca and Pino, Carmelo and Rundo, Francesco and Palazzo, Simone and Spampinato, Concetto},
  journal={Computer Vision and Image Understanding},
  volume={235},
  pages={103773},
  year={2023},
  publisher={Elsevier}
}
```

---

## 📌 Disclaimer
This is an **unofficial implementation** and may differ from the original authors’ code and results.
