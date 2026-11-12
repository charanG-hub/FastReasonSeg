from pathlib import Path
from PIL import Image, ImageDraw
import csv
import numpy as np

ROOT = Path.home() / "Documents/rvt/data/DAVIS"
OUT = Path("docs/recovered_results/visual_qa")
OUT.mkdir(parents=True, exist_ok=True)

SEQUENCES = {
    "Bear": {
        "gt": ROOT / "Annotations/480p/bear",
        "frames": ROOT / "JPEGImages/480p/bear",
        "pred": Path("data/bear_benchmark"),
        "csv": Path("data/bear_benchmark/results.csv"),
        "count": 82,
    },
    "Car-turn": {
        "gt": ROOT / "Annotations/480p/car-turn",
        "frames": ROOT / "JPEGImages/480p/car-turn",
        "pred": Path("data/car_turn_benchmark"),
        "csv": Path("data/car_turn_benchmark/results.csv"),
        "count": 80,
    },
}

# Inspect the first, quarter, middle, three-quarter, and last frames.
def selected_frames(count):
    return sorted(set([
        0,
        count // 4,
        count // 2,
        (3 * count) // 4,
        count - 1,
    ]))

def color_mask(mask, color=(0, 255, 0)):
    arr = np.asarray(mask) > 0
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    rgb[arr] = color
    return Image.fromarray(rgb)

for name, cfg in SEQUENCES.items():
    with cfg["csv"].open(newline="") as f:
        rows = {r["frame"]: r for r in csv.DictReader(f)}

    tiles = []
    for idx in selected_frames(cfg["count"]):
        fid = f"{idx:05d}"
        frame_path = cfg["frames"] / f"{fid}.jpg"
        gt_path = cfg["gt"] / f"{fid}.png"
        pred_path = cfg["pred"] / f"{fid}_pred.png"

        if not all(p.exists() for p in [frame_path, gt_path, pred_path]):
            print(f"{name} {fid}: missing input; skipped")
            continue

        frame = Image.open(frame_path).convert("RGB")
        gt = np.asarray(Image.open(gt_path)) > 0
        pred = np.asarray(Image.open(pred_path)) > 0

        if gt.shape != pred.shape:
            print(f"{name} {fid}: shape mismatch; skipped")
            continue

        intersection = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        iou = intersection / union if union else 1.0

        # Original frame, ground-truth mask, predicted mask, and comparison.
        gt_rgb = np.asarray(frame).copy()
        gt_rgb[gt] = (
            0.5 * gt_rgb[gt].astype(np.float32)
            + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)

        pred_rgb = np.asarray(frame).copy()
        pred_rgb[pred] = (
            0.5 * pred_rgb[pred].astype(np.float32)
            + 0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)

        # Green = ground truth only; red = prediction only;
        # yellow = overlap.
        comparison = np.zeros((*gt.shape, 3), dtype=np.uint8)
        comparison[np.logical_and(gt, pred)] = [255, 255, 0]
        comparison[np.logical_and(gt, ~pred)] = [0, 255, 0]
        comparison[np.logical_and(pred, ~gt)] = [255, 0, 0]

        panels = [
            ("Original", frame),
            ("Ground truth", Image.fromarray(gt_rgb)),
            ("Prediction", Image.fromarray(pred_rgb)),
            ("Difference", Image.fromarray(comparison)),
        ]

        for title, img in panels:
            tiles.append((f"{name} | frame {fid} | {title} | IoU {iou:.4f}", img))

        print(
            f"{name} frame {fid}: "
            f"CSV IoU={rows.get(fid, {}).get('iou', 'missing')}, "
            f"recalculated IoU={iou:.6f}"
        )

    if not tiles:
        print(f"{name}: no contact sheet generated")
        continue

    tile_w, tile_h = 320, 220
    label_h = 30
    columns = 4
    rows_count = (len(tiles) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_w, rows_count * (tile_h + label_h)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for i, (label, img) in enumerate(tiles):
        img = img.copy()
        img.thumbnail((tile_w, tile_h))
        x = (i % columns) * tile_w
        y = (i // columns) * (tile_h + label_h)
        sheet.paste(img, (x, y + label_h))
        draw.text((x + 4, y + 4), label, fill="black")

    out_path = OUT / f"{name.lower().replace('-', '_')}_contact_sheet.png"
    sheet.save(out_path)
    print(f"Saved: {out_path.resolve()}")

print("\nVisual QA contact sheets created.")
