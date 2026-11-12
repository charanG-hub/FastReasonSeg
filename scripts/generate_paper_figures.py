"""Generate high-resolution publication-quality figures for FastReasonSeg-3D."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT_DIR / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
})


def plot_ablation_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

    configs = [
        "Baseline",
        "+ SDG",
        "+ ABR",
        "Full (Ours)",
    ]
    colors = ["#94a3b8", "#38bdf8", "#818cf8", "#10b981"]

    # 1. Referring Accuracy
    ref_acc = [28.0, 84.0, 28.0, 84.0]
    bars1 = axes[0].bar(configs, ref_acc, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.2)
    axes[0].set_title("Referring Accuracy (%)", fontweight="bold", pad=10)
    axes[0].set_ylim(0, 100)
    axes[0].axhline(28.0, color="#ef4444", linestyle="--", alpha=0.5, label="Baseline")
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    # 2. Boundary Contour Accuracy F@2px
    f2_acc = [21.30, 21.30, 34.24, 34.24]
    bars2 = axes[1].bar(configs, f2_acc, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.2)
    axes[1].set_title("Boundary Contour F@2px (%)", fontweight="bold", pad=10)
    axes[1].set_ylim(0, 45)
    axes[1].axhline(21.30, color="#ef4444", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    # 3. Mask IoU (J)
    iou_vals = [94.61, 94.61, 96.22, 96.22]
    bars3 = axes[2].bar(configs, iou_vals, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.2)
    axes[2].set_title("Mask IoU (Jaccard %)", fontweight="bold", pad=10)
    axes[2].set_ylim(90, 100)
    axes[2].axhline(94.61, color="#ef4444", linestyle="--", alpha=0.5)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "ablation_metrics_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_category_breakdown():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    categories = [
        "Depth + Spatial",
        "Depth Only",
        "Spatial Relative",
        "Compound 3D",
        "Spatial Only",
        "Occlusion Depth",
    ]
    baseline_scores = [20.0, 44.4, 50.0, 0.0, 25.0, 0.0]
    proposed_scores = [100.0, 77.8, 100.0, 50.0, 100.0, 100.0]

    y = np.arange(len(categories))
    height = 0.35

    rects1 = ax.barh(y - height/2, baseline_scores, height, label="Baseline FastReasonSeg", color="#94a3b8", edgecolor="#334155")
    rects2 = ax.barh(y + height/2, proposed_scores, height, label="FastReasonSeg-3D (Ours)", color="#10b981", edgecolor="#064e3b")

    ax.set_xlabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Referring Expression Disambiguation by Category", fontweight="bold", pad=12)
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontweight="medium")
    ax.set_xlim(0, 115)
    ax.legend(loc="lower right", frameon=True)

    for bar in rects1:
        w = bar.get_width()
        ax.text(w + 1.5, bar.get_y() + bar.get_height()/2.0, f"{w:.1f}%", va="center", ha="left", color="#475569", fontsize=9)

    for bar in rects2:
        w = bar.get_width()
        ax.text(w + 1.5, bar.get_y() + bar.get_height()/2.0, f"{w:.1f}%", va="center", ha="left", color="#065f46", fontweight="bold", fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "category_accuracy_breakdown.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_mask_refinement_visual():
    gt_path = ROOT_DIR / "data" / "truck_ground_truth_mask.png"
    base_path = ROOT_DIR / "data" / "truck_mask.png"
    ref_path = ROOT_DIR / "results" / "truck_adaptive_refined_mask.png"
    img_path = ROOT_DIR / "data" / "truck.jpg"

    if not gt_path.exists() or not base_path.exists() or not ref_path.exists():
        print("Mask files missing, skipping visual mask figure.")
        return

    gt = np.array(Image.open(gt_path).convert("L")) > 127
    base = np.array(Image.open(base_path).convert("L")) > 127
    ref = np.array(Image.open(ref_path).convert("L")) > 127
    img = np.array(Image.open(img_path).convert("RGB"))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300)

    # 1. Input Image
    axes[0].imshow(img)
    axes[0].set_title("(a) Input Image", fontweight="bold")
    axes[0].axis("off")

    # 2. Ground Truth Mask
    axes[1].imshow(img)
    axes[1].imshow(gt, cmap="Greens", alpha=0.55)
    axes[1].set_title("(b) Ground Truth Mask", fontweight="bold")
    axes[1].axis("off")

    # 3. Baseline Mask with Bleed Highlighted
    # Bleed = base & ~gt (False positives)
    bleed = base & ~gt
    axes[2].imshow(img)
    axes[2].imshow(base, cmap="Blues", alpha=0.45)
    axes[2].imshow(bleed, cmap="Reds", alpha=0.7)
    axes[2].set_title("(c) Baseline Mask (Red = Bleed)", fontweight="bold")
    axes[2].axis("off")

    # 4. Refined Mask (AdaptiveBoundaryRefiner)
    axes[3].imshow(img)
    axes[3].imshow(ref, cmap="Greens", alpha=0.55)
    axes[3].set_title("(d) Refined Mask (ABR - Ours)", fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    out_path = FIGURES_DIR / "mask_refinement_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_ablation_comparison()
    plot_category_breakdown()
    plot_mask_refinement_visual()
    print("All paper figures generated successfully.")
