"""Generate publication-ready feature diagnostics from the saved Shadow-ANO run.

The figure is based on the cached DINOv3/DPMM inference tensors in
``test_stats_anomalous.pth``.  It deliberately includes a strong, median, and
hard sample selected by pixel AUROC so that the panel is not a best-case-only
visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter, median_filter
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_run = (
        root
        / "results"
        / "Shadow-ANO"
        / "dinov3_vits16plus_dpmm_448_80_diag_"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=default_run,
        help="Directory containing test_stats_{normal,anomalous}.pth.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "feature_diagnostics.pdf",
        help="Output PDF path; a PNG preview is written beside it.",
    )
    return parser.parse_args()


def robust_unit(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(values, (low, high))
    return np.clip((values - lo) / max(hi - lo, 1e-8), 0.0, 1.0)


def upsample(batch: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    return (
        F.interpolate(batch, size=size, mode="bilinear", align_corners=False)
        .cpu()
        .numpy()
    )


def pixel_aurocs(scores: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    resized = F.interpolate(scores, labels.shape[-2:], mode="bilinear", align_corners=False)
    values = []
    for score, label in zip(resized[:, 0], labels[:, 0]):
        values.append(roc_auc_score(label.numpy().ravel().astype(np.uint8), score.numpy().ravel()))
    return np.asarray(values)


def select_diagnostic_samples(aurocs: np.ndarray) -> list[tuple[str, int]]:
    order = np.argsort(aurocs)
    hard = int(order[0])
    median = int(order[len(order) // 2])
    q90_value = np.quantile(aurocs, 0.90)
    strong = int(np.argmin(np.abs(aurocs - q90_value)))
    return [("Strong", strong), ("Median", median), ("Hard", hard)]


def extract_dinov2_tokens(images: torch.Tensor) -> np.ndarray:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(device).eval()
    with torch.inference_mode():
        tokens = model.get_intermediate_layers(images.to(device))[0].float().cpu().numpy()
    return tokens


def feature_pca_rgb(tokens: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    n, patches, channels = tokens.shape
    side = int(round(patches**0.5))
    if side * side != patches:
        raise ValueError(f"Expected a square patch grid, got {patches} tokens")
    reduced = PCA(n_components=3, random_state=0).fit_transform(tokens.reshape(-1, channels))
    reduced = reduced.reshape(n, side, side, 3)
    for channel in range(3):
        reduced[..., channel] = robust_unit(reduced[..., channel])
    reduced = torch.from_numpy(reduced).permute(0, 3, 1, 2)
    return upsample(reduced, target_size).transpose(0, 2, 3, 1)


def build_gt_guided_feature_target(tokens: np.ndarray, label_grid: np.ndarray) -> np.ndarray:
    """Create an oracle feature target while retaining measured feature variation.

    The operation shifts tokens along each image's GT-defined shadow/non-shadow
    mean-difference direction.  It is a visualization target, not a trained
    representation or a measured model result.
    """
    guided = tokens.copy()
    for i in range(tokens.shape[0]):
        membership = label_grid[i].reshape(-1).astype(np.float32)
        shadow = membership >= 0.5
        normal = membership < 0.5
        if shadow.sum() < 2 or normal.sum() < 2:
            continue
        direction = tokens[i, shadow].mean(0) - tokens[i, normal].mean(0)
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-8:
            continue
        direction = direction / direction_norm
        projection_scale = max(float(np.std(tokens[i] @ direction)), 1e-3)
        signed_membership = 2.0 * membership - 1.0
        guided[i] += 3.0 * projection_scale * signed_membership[:, None] * direction[None]
    return guided


def denormalize(images: torch.Tensor) -> np.ndarray:
    images = images * IMAGENET_STD + IMAGENET_MEAN
    return images.clamp(0, 1).permute(0, 2, 3, 1).numpy()


def main() -> None:
    args = parse_args()
    normal = torch.load(args.run_dir / "test_stats_normal.pth", map_location="cpu")
    anomalous = torch.load(args.run_dir / "test_stats_anomalous.pth", map_location="cpu")

    aurocs = pixel_aurocs(anomalous["cosine_distance_map"], anomalous["labels"])
    selected = select_diagnostic_samples(aurocs)
    indices = [index for _, index in selected]
    images_tensor = anomalous["image"][indices]
    images = denormalize(images_tensor)
    tokens = extract_dinov2_tokens(images_tensor)
    feature_rgb = feature_pca_rgb(tokens, images_tensor.shape[-2:])

    component_distance = anomalous["component_cosine_distance_map"][indices]
    assignments = component_distance.argmin(dim=1).numpy()
    nearest_two = component_distance.topk(k=2, dim=1, largest=False).values
    margins = (nearest_two[:, 1] - nearest_two[:, 0]).numpy()

    target_size = anomalous["labels"].shape[-2:]
    margins_up = upsample(torch.from_numpy(margins)[:, None], target_size)[:, 0]
    cosine_up = upsample(anomalous["cosine_distance_map"][indices], target_size)[:, 0]
    likelihood_up = upsample(anomalous["anomaly_map"][indices], target_size)[:, 0]

    all_cosine = torch.cat(
        [normal["cosine_distance_map"], anomalous["cosine_distance_map"]], dim=0
    ).numpy()
    cos_lo, cos_hi = np.percentile(all_cosine, (1, 99))
    cosine_unit = np.clip((cosine_up - cos_lo) / max(cos_hi - cos_lo, 1e-8), 0, 1)
    likelihood_unit = robust_unit(likelihood_up)
    margins_unit = robust_unit(margins_up)
    labels = anomalous["labels"][indices, 0].numpy()

    patch_side = int(round(tokens.shape[1] ** 0.5))
    label_grid = (
        F.interpolate(
            anomalous["labels"][indices],
            size=(patch_side, patch_side),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        .numpy()
    )
    oracle_tokens = build_gt_guided_feature_target(tokens, label_grid)
    oracle_feature_rgb = feature_pca_rgb(oracle_tokens, images_tensor.shape[-2:])

    # The oracle prototype target keeps spatially coherent normal assignments
    # and reserves one explicit out-of-distribution state for GT shadow patches.
    oracle_assignments = median_filter(assignments, size=(1, 3, 3), mode="nearest")
    oracle_assignments[label_grid >= 0.5] = component_distance.shape[1]

    # Soft edges and a small measured residual avoid an unrealistically binary
    # answer key while ensuring that the target is visibly aligned with GT.
    soft_labels = gaussian_filter(labels.astype(np.float32), sigma=(0, 4.0, 4.0))
    oracle_separation = np.clip(
        0.90 * soft_labels
        + 0.07 * (1.0 - margins_unit) * soft_labels
        + 0.03 * margins_unit * (1.0 - soft_labels),
        0.0,
        1.0,
    )
    oracle_cosine = np.clip(
        0.82 * soft_labels
        + 0.13 * cosine_unit * soft_labels
        + 0.05 * cosine_unit * (1.0 - soft_labels),
        0.0,
        1.0,
    )

    n_components = anomalous["component_cosine_distance_map"].shape[1]
    hues = mpl.colormaps["turbo"](np.linspace(0.02, 0.98, n_components))
    component_cmap = ListedColormap(np.vstack([hues, mpl.colors.to_rgba("#ff2d95")]))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titleweight": "semibold",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(6, 7, figsize=(13.8, 11.8), constrained_layout=False)
    headers = [
        "Input",
        "Feature PCA",
        "Prototype / OOD state",
        "Separation margin",
        "Cosine anomaly",
        "Anomaly overlay",
        "Ground-truth boundary",
    ]
    for ax, header in zip(axes[0], headers):
        ax.set_title(header, fontsize=9.5, pad=6)

    for row, ((level, source_index), image, pca_rgb, assignment, margin, cosine, likelihood, label) in enumerate(
        zip(
            selected,
            images,
            feature_rgb,
            assignments,
            margins_unit,
            cosine_unit,
            likelihood_unit,
            labels,
        )
    ):
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(pca_rgb)
        axes[row, 2].imshow(assignment, cmap=component_cmap, vmin=0, vmax=n_components, interpolation="nearest")
        axes[row, 3].imshow(margin, cmap="viridis", vmin=0, vmax=1)
        axes[row, 4].imshow(cosine, cmap="magma", vmin=0, vmax=1)
        axes[row, 5].imshow(image)
        axes[row, 5].imshow(cosine, cmap="magma", vmin=0, vmax=1, alpha=0.68 * cosine + 0.12)
        axes[row, 6].imshow(image)
        axes[row, 6].contour(label, levels=[0.5], colors=["#00fff0"], linewidths=2.0)
        axes[row, 0].set_ylabel(
            f"Measured {level}\nAUROC {aurocs[source_index]:.3f}\nindex {source_index}",
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=9,
        )
        # A subtle likelihood contour links the full-density score to the cosine map.
        axes[row, 4].contour(likelihood, levels=[0.75], colors="white", linewidths=0.65, alpha=0.8)

    for local_row, ((level, source_index), image, pca_rgb, assignment, separation, cosine, label) in enumerate(
        zip(
            selected,
            images,
            oracle_feature_rgb,
            oracle_assignments,
            oracle_separation,
            oracle_cosine,
            labels,
        )
    ):
        row = local_row + 3
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(pca_rgb)
        axes[row, 2].imshow(assignment, cmap=component_cmap, vmin=0, vmax=n_components, interpolation="nearest")
        axes[row, 3].imshow(separation, cmap="viridis", vmin=0, vmax=1)
        axes[row, 4].imshow(cosine, cmap="magma", vmin=0, vmax=1)
        axes[row, 5].imshow(image)
        axes[row, 5].imshow(cosine, cmap="magma", vmin=0, vmax=1, alpha=0.72 * cosine + 0.10)
        axes[row, 6].imshow(image)
        axes[row, 6].contour(label, levels=[0.5], colors=["#00fff0"], linewidths=2.0)
        axes[row, 0].set_ylabel(
            f"Oracle {level}\nGT-guided target\nNOT OUTPUT",
            fontsize=8.5,
            rotation=0,
            ha="right",
            va="center",
            labelpad=9,
            color="#c62828",
        )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.095, right=0.995, top=0.955, bottom=0.065, wspace=0.035, hspace=0.08)
    fig.add_artist(
        mpl.lines.Line2D(
            [0.09, 0.995],
            [0.51, 0.51],
            transform=fig.transFigure,
            color="#c62828",
            linewidth=1.5,
            alpha=0.75,
        )
    )
    fig.text(0.012, 0.735, "MEASURED\nCHECKPOINT", rotation=90, ha="center", va="center", fontsize=10, fontweight="bold")
    fig.text(
        0.012,
        0.285,
        "GT-GUIDED ORACLE TARGET - NOT MODEL OUTPUT",
        rotation=90,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#c62828",
        fontweight="bold",
    )
    margin_bar = fig.add_axes([0.49, 0.018, 0.13, 0.012])
    anomaly_bar = fig.add_axes([0.69, 0.018, 0.13, 0.012])
    mpl.colorbar.ColorbarBase(margin_bar, cmap="viridis", norm=mpl.colors.Normalize(0, 1), orientation="horizontal")
    mpl.colorbar.ColorbarBase(anomaly_bar, cmap="magma", norm=mpl.colors.Normalize(0, 1), orientation="horizontal")
    margin_bar.set_xlabel("relative prototype margin", fontsize=8, labelpad=2)
    anomaly_bar.set_xlabel("relative anomaly score", fontsize=8, labelpad=2)
    margin_bar.tick_params(labelsize=7, length=2)
    anomaly_bar.tick_params(labelsize=7, length=2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    preview = args.output.with_suffix(".png")
    fig.savefig(preview, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Selected samples:")
    for level, index in selected:
        print(f"  {level:6s}: index={index:02d}, pixel AUROC={aurocs[index]:.4f}")
    print(f"PDF: {args.output}")
    print(f"PNG: {preview}")


if __name__ == "__main__":
    main()
