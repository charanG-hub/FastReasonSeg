from pathlib import Path
import csv
import json
from datetime import datetime

ROOT = Path.cwd()
OUT = ROOT / "docs" / "final_submission"
OUT.mkdir(parents=True, exist_ok=True)

RESULTS = ROOT / "docs" / "recovered_results"

def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def read_text(path):
    return path.read_text(encoding="utf-8") if path.exists() else "NOT AVAILABLE"

def fmt(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (ValueError, TypeError):
        return "N/A"

paper = read_csv(RESULTS / "paper_rvtbench_reference_values.csv")
sequences = read_csv(RESULTS / "final_sequence_summary.csv")
truck = read_csv(RESULTS / "truck_mask_metrics.csv")

status = read_text(RESULTS / "FINAL_PROJECT_STATUS.md")
audit = read_text(RESULTS / "MISSING_TABLES_AUDIT.md")
benchmark = read_text(RESULTS / "COMBINED_BENCHMARK_REPORT.md")

config_path = ROOT / "models" / "final_adapter" / "adapter_config.json"
config = {}
if config_path.exists():
    config = json.loads(config_path.read_text(encoding="utf-8"))

# Export the paper values exactly as stored.
paper_out = OUT / "paper_reference_values.csv"
if paper:
    with paper_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(paper[0].keys()))
        writer.writeheader()
        writer.writerows(paper)

# Build an explicit comparison table without mixing incompatible metrics.
comparison = []
for row in paper:
    comparison.append({
        "paper_metric": row.get("metric", ""),
        "paper_method": row.get("method", ""),
        "level_1_paper": row.get("level_1", ""),
        "level_2_paper": row.get("level_2", ""),
        "level_3_paper": row.get("level_3", ""),
        "our_official_RVTBench_result": "NOT EVALUATED / NOT VERIFIED",
        "comparison_status": "NOT DIRECTLY COMPARABLE",
        "reason": "Local DAVIS prototype results are not official RVTBench results."
    })

comparison_out = OUT / "paper_vs_project_comparison.csv"
if comparison:
    with comparison_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)

# Compose the report.
lines = []
lines += [
    "# FastReasonSeg — Research Paper Comparison Report",
    "",
    f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
    "",
    "## 1. Executive summary",
    "",
    "This project contains a local segmentation prototype, saved evaluation records, "
    "a LoRA adapter configuration, and paper-reported reference values.",
    "",
    "**Official RVTBench reproduction: NOT VERIFIED.**",
    "",
    "The local results below must not be presented as official FastReasonSeg paper "
    "results. The local benchmark sequences and the paper's RVTBench evaluation "
    "have not been established as the same dataset and protocol.",
    "",
    "## 2. Paper-reported reference results",
    "",
    "The following values are reproduced from the project's saved reference CSV. "
    "They have not been independently re-measured by this script.",
    "",
    "| Metric | Method | Level 1 | Level 2 | Level 3 |",
    "|---|---|---:|---:|---:|"
]

for r in paper:
    lines.append(
        f"| {r.get('metric','')} | {r.get('method','')} | "
        f"{r.get('level_1','')} | {r.get('level_2','')} | "
        f"{r.get('level_3','')} |"
    )

lines += [
    "",
    "## 3. Our local prototype results",
    "",
    "| Sequence | Frames | Mean IoU | Mean Dice | Seconds/frame |",
    "|---|---:|---:|---:|---:|"
]

for r in sequences:
    lines.append(
        f"| {r.get('sequence','')} | {r.get('frames','')} | "
        f"{r.get('mean_iou','')} | {r.get('mean_dice','')} | "
        f"{r.get('mean_seconds_per_frame','')} |"
    )

lines += [
    "",
    "The previously consolidated local records contain 162 valid IoU values, "
    "with mean IoU 0.971075; 162 valid Dice values, with mean Dice 0.985294; "
    "and 162 timing values, averaging 1.436 seconds per frame.",
    "",
    "These are arithmetic summaries of saved local CSV records. They do not "
    "independently validate annotations, mask alignment, or metric implementation.",
    "",
    "## 4. Truck single-image mask evaluation",
    "",
    "| Prediction mask | IoU | Dice | Boundary F (2 px) |",
    "|---|---:|---:|---:|"
]

for r in truck:
    lines.append(
        f"| {r.get('prediction_mask','')} | {r.get('iou','')} | "
        f"{r.get('dice','')} | {r.get('boundary_f_2px','')} |"
    )

lines += [
    "",
    "Truck results are from a separate available single-image annotation. "
    "Boundary F uses a 2-pixel tolerance and is not an RVTBench result.",
    "",
    "## 5. Direct comparison status",
    "",
    "| Comparison | Status | Explanation |",
    "|---|---|---|",
    "| Paper RVTBench J vs local DAVIS IoU | Not directly comparable | Different or unverified dataset/protocol |",
    "| Paper RVTBench F vs local truck boundary F | Not directly comparable | Different evaluation and tolerance protocol |",
    "| Paper model FPS vs local seconds/frame | Not established | Hardware and timing protocol are not matched |",
    "| Official RVTBench levels 1–3 | Not verified | Official test data and level assignments not established |",
    "",
    "## 6. Adapter configuration found in repository",
    "",
    f"- Base model: `{config.get('base_model_name_or_path', 'Unknown')}`",
    f"- Adapter type: `{config.get('peft_type', 'Unknown')}`",
    f"- LoRA rank: `{config.get('r', 'Unknown')}`",
    f"- LoRA alpha: `{config.get('lora_alpha', 'Unknown')}`",
    f"- Target modules: `{', '.join(config.get('target_modules', []))}`",
    f"- Adapter weights present: `{'Yes' if (ROOT / 'models/final_adapter/adapter_model.safetensors').exists() else 'No'}`",
    "",
    "The presence of adapter weights and configuration does not by itself prove "
    "that the full paper training or distillation pipeline was reproduced.",
    "",
    "## 7. Missing work and limitations",
    "",
    "- Official RVTBench evaluation is not verified.",
    "- The existing mask audit found zero usable matched GT/prediction pairs "
    "for independently generating new J/F scores.",
    "- Full paper-specific teacher training, SFT/GRPO training and student "
    "distillation are reported as unimplemented.",
    "- Existing video results do not establish persistent tracking.",
    "- Only a small number of local sequences are represented.",
    "- The dependency environment reports a `datasets` / `huggingface-hub` conflict.",
    "",
    "## 8. Reproducibility commands",
    "",
    "```bash",
    "cd ~/Documents/FastReasonSeg",
    "source .venv/bin/activate",
    "python finalize_fastreasonseg.py --root .",
    "python build_research_comparison.py",
    "```",
    "",
    "## 9. Source files used",
    "",
    "- `docs/recovered_results/paper_rvtbench_reference_values.csv`",
    "- `docs/recovered_results/final_sequence_summary.csv`",
    "- `docs/recovered_results/truck_mask_metrics.csv`",
    "- `docs/recovered_results/FINAL_PROJECT_STATUS.md`",
    "- `docs/recovered_results/MISSING_TABLES_AUDIT.md`",
    "- `docs/recovered_results/COMBINED_BENCHMARK_REPORT.md`",
    "",
    "## 10. Academic integrity statement",
    "",
    "This report separates paper-reported reference values from locally measured "
    "prototype results. Missing official benchmark results are explicitly marked "
    "as not evaluated or not verified. No missing scores have been fabricated.",
]

report = OUT / "FINAL_RESEARCH_COMPARISON.md"
report.write_text("\n".join(lines) + "\n", encoding="utf-8")

# Save an easy-to-read text summary.
summary = OUT / "DEMO_SUMMARY.txt"
summary.write_text(
    "FASTREASONSEG PROJECT DEMONSTRATION\n"
    "===================================\n"
    "Local combined mean IoU: 97.1075%\n"
    "Local combined mean Dice: 98.5294%\n"
    "Local mean processing time: 1.436 seconds/frame\n"
    "Official RVTBench: NOT VERIFIED\n"
    "Paper training/distillation pipeline: NOT COMPLETE\n"
    "See FINAL_RESEARCH_COMPARISON.md for full qualifications.\n",
    encoding="utf-8"
)

print("\nRESEARCH COMPARISON PACKAGE CREATED")
print("===================================")
print(f"Report: {report}")
print(f"Paper references: {paper_out}")
print(f"Comparison CSV: {comparison_out}")
print(f"Demo summary: {summary}")
print(f"Paper reference rows: {len(paper)}")
print(f"Local sequence rows: {len(sequences)}")
print(f"Truck mask rows: {len(truck)}")
print("\nOfficial RVTBench status remains NOT VERIFIED.")
