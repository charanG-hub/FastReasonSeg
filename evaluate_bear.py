from pathlib import Path
import csv
import time
import traceback

import numpy as np
import torch
from PIL import Image

from app import detect_objects, segment_image

# DAVIS dataset
ROOT = Path.home() / "Documents/rvt/data/DAVIS"
FRAMES_DIR = ROOT / "JPEGImages/480p/bear"
MASKS_DIR = ROOT / "Annotations/480p/bear"

# Start with three frames to validate the complete pipeline.
FRAME_IDS = [f"{i:05d}" for i in range(82)]

PROMPT = "bear"
THRESHOLD = 0.05

OUTPUT_DIR = Path("data/bear_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Device:", (
    "MPS" if torch.backends.mps.is_available()
    else "CUDA" if torch.cuda.is_available()
    else "CPU"
))
print("Prompt:", PROMPT)
print("Frames:", FRAME_IDS)

def calculate_metrics(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Mask shape mismatch: prediction {pred.shape}, "
            f"ground truth {gt.shape}"
        )

    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())

    pred_pixels = int(pred.sum())
    gt_pixels = int(gt.sum())

    iou = intersection / union if union else 1.0
    dice_denominator = pred_pixels + gt_pixels
    dice = (
        2 * intersection / dice_denominator
        if dice_denominator else 1.0
    )

    return iou, dice, pred_pixels, gt_pixels

results = []

for frame_id in FRAME_IDS:
    image_path = FRAMES_DIR / f"{frame_id}.jpg"
    mask_path = MASKS_DIR / f"{frame_id}.png"

    print(f"\nEvaluating frame {frame_id}...")

    if not image_path.exists() or not mask_path.exists():
        print("SKIP: image or ground-truth mask missing.")
        continue

    image = Image.open(image_path).convert("RGB")
    gt_image = Image.open(mask_path)
    gt = np.array(gt_image) > 0

    start = time.perf_counter()

    try:
        boxes, scores = detect_objects(
            image, PROMPT, threshold=THRESHOLD
        )

        if len(boxes) == 0:
            print("No object detected.")
            results.append({
                "frame": frame_id,
                "detected": False,
                "confidence": "",
                "iou": 0.0,
                "dice": 0.0,
                "predicted_pixels": 0,
                "ground_truth_pixels": int(gt.sum()),
                "elapsed_seconds": round(
                    time.perf_counter() - start, 3
                ),
                "error": "No object detected",
            })
            continue

        # Use the highest-confidence detection.
        index = int(np.argmax(scores))
        box = boxes[index]
        confidence = float(scores[index])

        pred, sam_score = segment_image(image, box)
        pred = np.asarray(pred, dtype=bool)

        iou, dice, pred_pixels, gt_pixels = calculate_metrics(
            pred, gt
        )

        elapsed = time.perf_counter() - start

        # Save predicted mask and visual overlay.
        Image.fromarray(
            (pred.astype(np.uint8) * 255)
        ).save(OUTPUT_DIR / f"{frame_id}_pred.png")

        overlay = np.array(image).copy()
        overlay[pred] = (
            0.5 * overlay[pred].astype(np.float32)
            + 0.5 * np.array([0, 255, 0], dtype=np.float32)
        ).astype(np.uint8)
        Image.fromarray(overlay).save(
            OUTPUT_DIR / f"{frame_id}_overlay.png"
        )

        result = {
            "frame": frame_id,
            "detected": True,
            "confidence": round(confidence, 5),
            "sam_score": round(float(sam_score), 5),
            "iou": round(iou, 6),
            "dice": round(dice, 6),
            "predicted_pixels": pred_pixels,
            "ground_truth_pixels": gt_pixels,
            "elapsed_seconds": round(elapsed, 3),
            "error": "",
        }
        results.append(result)

        print("  Detection confidence:", round(confidence, 4))
        print("  IoU:", round(iou, 4))
        print("  Dice:", round(dice, 4))
        print("  Time (seconds):", round(elapsed, 2))

    except Exception as exc:
        print("ERROR:", repr(exc))
        traceback.print_exc()
        results.append({
            "frame": frame_id,
            "detected": False,
            "confidence": "",
            "iou": "",
            "dice": "",
            "predicted_pixels": "",
            "ground_truth_pixels": int(gt.sum()),
            "elapsed_seconds": round(
                time.perf_counter() - start, 3
            ),
            "error": repr(exc),
        })

csv_path = OUTPUT_DIR / "results.csv"

fieldnames = sorted({
    key for result in results for key in result
})

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

valid = [
    r for r in results
    if r.get("detected") is True
    and isinstance(r.get("iou"), (float, int))
]

print("\n========== SUMMARY ==========")
print("Frames attempted:", len(FRAME_IDS))
print("Frames with predictions:", len(valid))

if valid:
    print("Mean IoU:", round(
        float(np.mean([r["iou"] for r in valid])), 4
    ))
    print("Mean Dice:", round(
        float(np.mean([r["dice"] for r in valid])), 4
    ))
    print(
        "NOTE: These means cover detected frames only. "
        "Missed detections are reported separately."
    )

print("CSV:", csv_path.resolve())
print("Output directory:", OUTPUT_DIR.resolve())
