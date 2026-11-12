import csv
from pathlib import Path
from statistics import mean
from datetime import datetime

ROOT = Path(".")
OUTPUT = ROOT / "docs" / "recovered_results"
OUTPUT.mkdir(parents=True, exist_ok=True)

BENCHMARKS = {
    "Bear": ROOT / "data" / "bear_benchmark" / "results.csv",
    "Car-turn": ROOT / "data" / "car_turn_benchmark" / "results.csv",
}

def load_rows(path):
    if not path.exists():
        print(f"WARNING: Missing results file: {path}")
        return []

    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def numeric_values(rows, key):
    values = []
    for row in rows:
        try:
            value = float(row[key])
            if value == value:
                values.append(value)
        except (KeyError, TypeError, ValueError):
            pass
    return values

all_rows = []
sequence_stats = []

for name, path in BENCHMARKS.items():
    rows = load_rows(path)

    if not rows:
        continue

    ious = numeric_values(rows, "iou")
    dices = numeric_values(rows, "dice")
    times = numeric_values(rows, "elapsed_seconds")
    detected = sum(
        row.get("detected", "").lower() == "true"
        for row in rows
    )
    errors = sum(bool(row.get("error", "").strip()) for row in rows)

    stats = {
        "sequence": name,
        "frames": len(rows),
        "detections": detected,
        "mean_iou": mean(ious) if ious else None,
        "mean_dice": mean(dices) if dices else None,
        "mean_seconds_per_frame": mean(times) if times else None,
        "frames_with_errors": errors,
        "metric_rows": min(len(ious), len(dices)),
    }

    sequence_stats.append(stats)

    for row in rows:
        combined = dict(row)
        combined["sequence"] = name
        all_rows.append(combined)

if not sequence_stats:
    raise SystemExit("No benchmark CSVs were found. Check the paths above.")

combined_csv = OUTPUT / "combined_benchmark_results.csv"

fieldnames = ["sequence"] + list(dict.fromkeys(
    key for row in all_rows for key in row if key != "sequence"
))

with combined_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

# Compute combined means from individual frame records.
all_ious = numeric_values(all_rows, "iou")
all_dices = numeric_values(all_rows, "dice")
all_times = numeric_values(all_rows, "elapsed_seconds")

report = []
report.append("# Combined Segmentation Benchmark Report")
report.append("")
report.append(f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}")
report.append("")
report.append("> These are prototype results on DAVIS sequences. They are not official RVTBench results and do not establish a full reproduction of the FastReasonSeg paper.")
report.append("")
report.append("## Per-sequence results")
report.append("")
report.append("| Sequence | Frames | Detections | Mean IoU | Mean Dice | Mean seconds/frame | Errors |")
report.append("|---|---:|---:|---:|---:|---:|---:|")

for s in sequence_stats:
    iou = f"{s['mean_iou']:.6f}" if s["mean_iou"] is not None else "N/A"
    dice = f"{s['mean_dice']:.6f}" if s["mean_dice"] is not None else "N/A"
    timing = f"{s['mean_seconds_per_frame']:.3f}" if s["mean_seconds_per_frame"] is not None else "N/A"
    report.append(
        f"| {s['sequence']} | {s['frames']} | {s['detections']} | "
        f"{iou} | {dice} | {timing} | {s['frames_with_errors']} |"
    )

report.append("")
report.append("## Combined results")
report.append("")
report.append(f"- Total frame records: {len(all_rows)}")
report.append(f"- Rows with IoU values: {len(all_ious)}")
report.append(f"- Rows with Dice values: {len(all_dices)}")
report.append(f"- Rows with timing values: {len(all_times)}")
report.append(f"- Combined mean IoU: {mean(all_ious):.6f}" if all_ious else "- Combined mean IoU: N/A")
report.append(f"- Combined mean Dice: {mean(all_dices):.6f}" if all_dices else "- Combined mean Dice: N/A")
report.append(f"- Combined mean processing time: {mean(all_times):.3f} seconds/frame" if all_times else "- Combined mean processing time: N/A")
report.append("")
report.append("## Interpretation and limitations")
report.append("")
report.append("- IoU and Dice values are calculated from the saved per-frame CSV records.")
report.append("- Combined metrics use individual frame values, rather than averaging sequence means.")
report.append("- File counts and CSV values alone do not independently validate mask alignment or metric implementation.")
report.append("- These runs use independent frame-level detection and segmentation; they should not be described as persistent video tracking unless that capability is separately verified.")
report.append("- Two sequences are a limited evaluation and are not sufficient to establish generalization or state-of-the-art performance.")
report.append("- Verify the evaluation script, annotation alignment, and mask thresholding before treating these numbers as final scientific results.")

report_path = OUTPUT / "COMBINED_BENCHMARK_REPORT.md"
report_path.write_text("\n".join(report) + "\n")

print("\n=== COMBINED BENCHMARK SUMMARY ===")
for s in sequence_stats:
    print(
        f"{s['sequence']}: frames={s['frames']}, "
        f"IoU={s['mean_iou']}, Dice={s['mean_dice']}, "
        f"errors={s['frames_with_errors']}"
    )

print("\nTotal frame records:", len(all_rows))
print("Combined mean IoU:", f"{mean(all_ious):.6f}" if all_ious else "N/A")
print("Combined mean Dice:", f"{mean(all_dices):.6f}" if all_dices else "N/A")
print("Combined mean time/frame:", f"{mean(all_times):.3f}s" if all_times else "N/A")
print("\nReport:", report_path.resolve())
print("Combined CSV:", combined_csv.resolve())
