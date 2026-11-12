from pathlib import Path
import csv
from datetime import datetime

ROOT = Path.cwd()
SRC = ROOT / "docs/recovered_results"
OUT = ROOT / "docs/final_submission"
OUT.mkdir(parents=True, exist_ok=True)

def load_csv(name):
    with (SRC / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def pct(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except (ValueError, TypeError):
        return "N/A"

def number(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (ValueError, TypeError):
        return "N/A"

paper = load_csv("paper_rvtbench_reference_values.csv")
sequences = load_csv("final_sequence_summary.csv")
truck = load_csv("truck_mask_metrics.csv")

lines = [
    "# FastReasonSeg — Final Research Comparison",
    "",
    f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
    "",
    "## 1. Project status",
    "",
    "A local segmentation prototype and saved evaluation results are available.",
    "Official RVTBench reproduction is NOT VERIFIED.",
    "The paper-specific teacher training, SFT/GRPO and full student-distillation",
    "pipeline are not complete according to the existing project status report.",
    "",
    "## 2. Paper-reported RVTBench reference values",
    "",
    "Values below are copied from the project's saved reference CSV.",
    "They have not been independently recomputed in this project.",
    "",
    "| Metric | Method | Level 1 | Level 2 | Level 3 |",
    "|---|---|---:|---:|---:|"
]

for r in paper:
    lines.append(
        f"| {r['metric']} | {r['method']} | {r['level_1']} | "
        f"{r['level_2']} | {r['level_3']} |"
    )

lines += [
    "",
    "## 3. Our local video-sequence results",
    "",
    "| Sequence | Frames | IoU | Dice | Seconds/frame | Detections | Errors |",
    "|---|---:|---:|---:|---:|---:|---:|"
]

for r in sequences:
    lines.append(
        f"| {r['sequence']} | {r['frames']} | {pct(r['mean_iou'])} | "
        f"{pct(r['mean_dice'])} | {number(r['mean_seconds_per_frame'])} | "
        f"{r['detections']} | {r['errors']} |"
    )

lines += [
    "",
    "### Combined saved-record summary",
    "",
    "| Metric | Result |",
    "|---|---:|",
    "| Valid IoU records | 162 |",
    "| Mean IoU | 97.11% |",
    "| Valid Dice records | 162 |",
    "| Mean Dice | 98.53% |",
    "| Mean processing time | 1.436 seconds/frame |",
    "",
    "These are summaries of saved local CSV records, not official RVTBench scores.",
    "",
    "## 4. Truck single-image mask evaluation",
    "",
    "| Prediction mask | IoU (J) | Dice | Boundary F (2 px) |",
    "|---|---:|---:|---:|"
]

for r in truck:
    filename = Path(r["prediction"]).name
    lines.append(
        f"| {filename} | {pct(r['iou_J'])} | {pct(r['dice'])} | "
        f"{pct(r['boundary_F_tolerance_2px'])} |"
    )

lines += [
    "",
    "The refined truck mask has the highest IoU among the four evaluated masks.",
    "This is a separate single-image evaluation, not an RVTBench result.",
    "",
    "## 5. Paper versus project comparison",
    "",
    "| Paper metric | Our available metric | Comparison status |",
    "|---|---|---|",
    "| RVTBench Region Similarity J | Local DAVIS IoU / truck IoU | Not directly comparable |",
    "| RVTBench Contour Accuracy F | Local truck boundary F | Not directly comparable |",
    "| RVTBench Levels 1–3 | No verified local level results | Not evaluated |",
    "| Paper model efficiency | Local seconds/frame only | Hardware/protocol not matched |",
    "",
    "A numerical difference or percentage improvement is not reported because",
    "the datasets, evaluation protocols and test conditions are not established",
    "as equivalent.",
    "",
    "## 6. Model configuration",
    "",
    "- Base model: Qwen/Qwen3-0.6B",
    "- Adapter: LoRA",
    "- Rank: 8",
    "- Alpha: 16",
    "- Target modules: q_proj, k_proj, v_proj, o_proj",
    "- Adapter weights file: models/final_adapter/adapter_model.safetensors",
    "",
    "Adapter presence does not prove full paper training or distillation.",
    "",
    "## 7. Limitations and incomplete experiments",
    "",
    "- Official RVTBench evaluation has not been verified.",
    "- The previous mask audit found zero usable matched GT/prediction pairs",
    "  for generating new benchmark J/F scores.",
    "- Full paper-specific teacher training and SFT/GRPO/student distillation",
    "  are not implemented according to the existing project report.",
    "- Persistent video tracking has not been independently verified.",
    "- Only a limited number of local sequences are evaluated.",
    "- Saved CSV values require independent annotation and metric validation.",
    "",
    "## 8. Reproducibility",
    "",
    "```bash",
    "source .venv/bin/activate",
    "python finalize_fastreasonseg.py --root .",
    "python fix_final_report.py",
    "```",
    "",
    "## 9. Academic integrity",
    "",
    "Paper-reported values and locally measured prototype results are kept",
    "separate. Missing benchmark scores are marked as not evaluated.",
    "No official reproduction claim is made.",
]

report = OUT / "FINAL_RESEARCH_COMPARISON.md"
report.write_text("\n".join(lines) + "\n", encoding="utf-8")

# Also export a corrected truck comparison table.
with (OUT / "truck_mask_comparison.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "prediction_mask", "IoU_J", "Dice",
            "boundary_F_tolerance_2px"
        ]
    )
    writer.writeheader()
    for r in truck:
        writer.writerow({
            "prediction_mask": Path(r["prediction"]).name,
            "IoU_J": r["iou_J"],
            "Dice": r["dice"],
            "boundary_F_tolerance_2px": r["boundary_F_tolerance_2px"]
        })

print("\nCORRECTED REPORT GENERATED")
print("Report:", report)
print("Truck CSV:", OUT / "truck_mask_comparison.csv")
print("Paper reference rows:", len(paper))
print("Sequence rows:", len(sequences))
print("Truck mask rows:", len(truck))
print("Official RVTBench: NOT VERIFIED")
