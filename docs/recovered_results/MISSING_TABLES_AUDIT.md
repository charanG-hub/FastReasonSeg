# FastReasonSeg Missing Tables — Automated Audit

- Repository root: `/Users/charankumar/Documents/FastReasonSeg`
- Audit timestamp: 2026-10-30T15:29:19+05:30
- Boundary tolerance: 2 pixels

## Result

- Mask files discovered in repository: **654**
- Ground-truth mask files selected: **0**
- Predicted mask files selected: **0**
- Matching GT/prediction pairs: **0**
- Pairs successfully evaluated: **0**
- Evaluation errors: **0**

## Table status

| Table | Status |
|---|---|
| Bear DAVIS IoU/Dice/timing | Existing report found; see combined benchmark report |
| Car-turn DAVIS IoU/Dice/timing | Existing report found; see combined benchmark report |
| Combined DAVIS summary | Existing report found; not official RVTBench |
| Region Similarity (J) from discovered mask pairs | NOT AVAILABLE — no usable matched mask pairs found |
| Contour Accuracy (F) from discovered mask pairs | NOT AVAILABLE — no usable matched mask pairs found |
| Official RVTBench Level 1–3 table | NOT VERIFIED — official RVTBench data and level labels not established |

Important: J and F computed here are only valid for the discovered GT/prediction mask pairs. This script does not convert DAVIS scores into RVTBench scores and does not claim the pairs are official RVTBench samples.

## Discovered mask directories

| Directory | Classification | Mask files |
|---|---|---:|
| `docs/recovered_results/visual_qa` | mask_directory_unclassified | 4 |
| `data` | mask_directory_unclassified | 21 |
| `data/car_turn_benchmark` | mask_directory_unclassified | 160 |
| `data/dog_video_frames` | mask_directory_unclassified | 289 |
| `data/video_test_frames` | mask_directory_unclassified | 5 |
| `data/bear_benchmark` | mask_directory_unclassified | 165 |
| `data/bear_evaluation` | mask_directory_unclassified | 10 |

## Grouped J/F results

No valid paired masks were found, so no new J/F scores were calculated.

## Existing relevant result files

- `data/bear_benchmark/bear_benchmark_summary.csv`
- `data/bear_benchmark/results.csv`
- `data/bear_benchmark/results_82_frames.csv`
- `data/bear_evaluation/results.csv`
- `data/car_turn_benchmark/results.csv`
- `data/mask_comparison_metrics.csv`
- `data/query_plan_results.json`
- `data/truck_evaluation_results.json`
- `data/video_segmentation_metrics.csv`
- `docs/FINAL_EVALUATION_REPORT.md`
- `docs/recovered_results/COMBINED_BENCHMARK_REPORT.md`
- `docs/recovered_results/FINAL_BEAR_BENCHMARK_REPORT.md`
- `docs/recovered_results/FINAL_EVALUATION_REPORT.md`
- `docs/recovered_results/combined_benchmark_results.csv`
- `docs/recovered_results/evaluation/video_segmentation_metrics.csv`
- `video_segmentation_metrics.csv`

## Files created

- Per-frame J/F CSV not created because no valid mask pairs were available.
- Grouped J/F CSV not created because no valid mask pairs were available.
- `docs/recovered_results/MISSING_TABLES_AUDIT.md` — this audit report
