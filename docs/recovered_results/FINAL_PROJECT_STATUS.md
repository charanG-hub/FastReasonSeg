# FastReasonSeg Project Finalization Report

Generated: 2026-11-11T09:36:57+05:30

> Scope: consolidate existing saved CSV metrics and evaluate available single-image GT/prediction masks. This script does not run model inference, train models, or certify an official RVTBench evaluation.

## 1. Existing per-sequence results

| Sequence | CSV rows | IoU rows | Mean IoU | Median IoU | Dice rows | Mean Dice | Mean seconds/frame | Detections | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bear | 82 | 82 | 0.966016 | 0.966693 | 82 | 0.982706 | 1.379 | 82 | 0 |
| Car-turn | 80 | 80 | 0.976261 | 0.980697 | 80 | 0.987947 | 1.494 | 80 | 0 |
| Bear-evaluation | 5 | 0 | N/A | N/A | 0 | N/A | N/A | 5 | 0 |

## 2. Combined saved-record summary

- Sequences with readable results: **3**
- Total CSV rows: **167**
- Valid saved IoU values: **162**; mean: **0.971075**
- Valid saved Dice values: **162**; mean: **0.985294**
- Valid timing values: **162**; mean: **1.436 seconds/frame**

These are arithmetic summaries of existing CSV values. They do not independently verify the original annotations, mask alignment, or metric implementation.

## 3. Truck single-image mask evaluation

| Prediction mask | IoU / J | Dice | Boundary F (2 px) | GT pixels | Predicted pixels |
|---|---:|---:|---:|---:|---:|
| truck_sam2_refined_mask.png | 0.854482 | 0.921532 | 0.121630 | 644003 | 571612 |
| truck_refined_mask.png | 0.950282 | 0.974507 | 0.252281 | 644003 | 639940 |
| truck_clean_mask.png | 0.857372 | 0.923210 | 0.133170 | 644003 | 573865 |
| truck_mask.png | 0.946139 | 0.972324 | 0.228597 | 644003 | 635969 |

Boundary F is calculated with a 2-pixel tolerance on the available truck annotation and is not an RVTBench result.
## 4. Paper reference table

Reference values supplied by the project owner are saved to `paper_rvtbench_reference_values.csv`. They are paper-reported RVTBench values and are not measurements of this local prototype. Local DAVIS results must not be compared as if they were RVTBench results without matching dataset, levels, prompts, and evaluation protocol.

## 5. Implementation status and honest limitations

- Existing code/results demonstrate a segmentation prototype with saved frame metrics and motion/mask analyses.
- Existing reports describe frame-wise detection/segmentation; do not claim persistent tracking unless the relevant implementation is verified.
- A small number of video sequences does not establish generalization.
- The full paper-specific teacher training, SFT/GRPO training, and student distillation pipeline remains unimplemented according to the existing project report.
- This script does not fabricate missing metrics and does not label prototype results as official RVTBench.

## 6. Generated files

- `final_per_frame_results.csv` — merged per-frame CSV records
- `final_sequence_summary.csv` — sequence-level statistics
- `truck_mask_metrics.csv` — GT-based truck mask IoU, Dice, and 2-pixel boundary F if masks exist
- `paper_rvtbench_reference_values.csv` — paper reference values supplied by the project owner
- `FINAL_PROJECT_STATUS.md` — this report

## 7. Reproducibility command

```bash
source .venv/bin/activate
python finalize_fastreasonseg.py --root .
```
