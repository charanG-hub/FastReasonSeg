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
    },
    "Car-turn": {
        "gt": ROOT / "Annotations/480p/car-turn",
        "frames": ROOT / "JPEGImages/480p/car-turn",
        "pred": Path("data/car_turn_benchmark"),
        "csv": Path("data/car_turn_benchmark/results.csv"),
    },
}

for name, cfg in SEQUENCES.items():
    with cfg["csv"].open(newline="") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r.get("iou") not in ("", None)]
    rows.sort(key=lambda r: float(r["iou"]))
    worst = rows[:5]

    tile_w, tile_h, label_h = 320, 200, 30
    columns = 4
    sheet = Image.new("RGB", (columns * tile_w, len(worst) * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for row_idx, row in enumerate(worst):
        fid = row["frame"]
        frame_path = cfg["frames"] / f"{fid}.jpg"
        gt_path = cfg["gt"] / f"{fid}.png"
        pred_path = cfg["pred"] / f"{fid}_pred.png"

        if not all(p.exists() for p in [frame_path, gt_path, pred_path]):
            print(f"{name} {fid}: missing file")
            continue

        frame = Image.open(frame_path).convert("RGB")
        gt = np.asarray(Image.open(gt_path)) > 0
        pred = np.asarray(Image.open(pred_path)) > 0

        if gt.shape != pred.shape:
            print(f"{name} {fid}: shape mismatch")
            continue

        original = np.asarray(frame).copy()

        gt_overlay = original.copy()
        gt_overlay[gt] = (
            0.5 * gt_overlay[gt].astype(np.float32)
            + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)

        pred_overlay = original.copy()
        pred_overlay[pred] = (
            0.5 * pred_overlay[pred].astype(np.float32)
            + 0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)

        diff = np.zeros((*gt.shape, 3), dtype=np.uint8)
        diff[np.logical_and(gt, pred)] = [255, 255, 0]
        diff[np.logical_and(gt, ~pred)] = [0, 255, 0]
        diff[np.logical_and(pred, ~gt)] = [255, 0, 0]

        panels = [
            ("Original", frame),
            ("Ground truth (green)", Image.fromarray(gt_overlay)),
            ("Prediction (red)", Image.fromarray(pred_overlay)),
            ("Difference (yellow=overlap)", Image.fromarray(diff)),
        ]

        for col, (title, img) in enumerate(panels):
            img = img.copy()
            img.thumbnail((tile_w, tile_h))
            x = col * tile_w
            y = row_idx * (tile_h + label_h)
            sheet.paste(img, (x, y + label_h))
            draw.text(
                (x + 4, y + 4),
                f"{fid} | IoU {float(row['iou']):.4f} | {title}",
                fill="black",
            )

        print(f"{name} frame {fid}: IoU={float(row['iou']):.4f}")

    out = OUT / f"{name.lower().replace('-', '_')}_failure_cases.png"
    sheet.save(out)
    print(f"Saved: {out.resolve()}")

print("\nFailure-case contact sheets created.")
