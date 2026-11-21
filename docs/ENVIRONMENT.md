# Environment

Tested on Linux (training) and Windows 11 (evaluation) with Python 3.10 and an NVIDIA RTX 4090 D.

## 1. Conda environment

```bash
conda create -n shadow python=3.10
conda activate shadow
```

## 2. PyTorch (CUDA 11.8)

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only machines, drop the `--index-url` flag.

## 3. Remaining packages

```bash
pip install -r requirements.txt
```

## Notes

- `anomalib==0.6.0` is pinned because the feature extractor and metrics APIs changed in later releases.
- `numpy==1.26.4` is required by the pinned `opencv-python` and `scikit-learn` wheels.
- The DINOv2 weights are loaded through `torch.hub` on first use and are cached in `~/.cache/torch`.
- Evaluation writes its outputs next to the checkpoints under `results/`; no external service is required when `wandb_log` is `false`.
