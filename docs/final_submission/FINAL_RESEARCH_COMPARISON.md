# FastReasonSeg — Final Research Comparison

Generated: 2026-11-11T09:42:41+05:30

## 1. Project status

A local segmentation prototype and saved evaluation results are available.
Official RVTBench reproduction is NOT VERIFIED.
The paper-specific teacher training, SFT/GRPO and full student-distillation
pipeline are not complete according to the existing project status report.

## 2. Paper-reported JiTBench reference values (Table 1)

**Correction:** The original project CSV was incorrectly labelled RVTBench. Its three-level values match the paper's JiTBench Table 1 averages across semantic, spatial, and temporal categories. They are retained as JiTBench values, not RVTBench values.

See `paper_jitbench_reference_values.csv` for the complete 26-row reference table. These are paper-reported values, not project measurements.

## 3. Paper-reported RVTBench reference values (Table 2)

RVTBench uses **four** difficulty levels. The table below reports the paper's average Region Similarity (J) and Contour Accuracy (F) across semantic, spatial, and temporal categories at each level. Complete values for all 13 methods are in `paper_rvtbench_reference_values.csv`.

| Method | J L1 | J L2 | J L3 | J L4 | F L1 | F L2 | F L3 | F L4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LISA-7B | 0.435 | 0.420 | 0.398 | 0.377 | 0.408 | 0.388 | 0.369 | 0.343 |
| LISA-13B | 0.466 | 0.452 | 0.431 | 0.411 | 0.444 | 0.426 | 0.405 | 0.387 |
| VISA-7B | 0.429 | 0.412 | 0.394 | 0.372 | 0.413 | 0.396 | 0.379 | 0.358 |
| VISA-13B | 0.481 | 0.462 | 0.443 | 0.418 | 0.463 | 0.445 | 0.427 | 0.402 |
| LLM-Seg | 0.130 | 0.105 | 0.103 | 0.097 | 0.168 | 0.143 | 0.140 | 0.134 |
| SegZero | 0.446 | 0.423 | 0.404 | 0.383 | 0.432 | 0.410 | 0.390 | 0.367 |
| CoReS-7B | 0.437 | 0.408 | 0.401 | 0.379 | 0.422 | 0.405 | 0.387 | 0.364 |
| CoReS-13B | 0.451 | 0.422 | 0.415 | 0.393 | 0.436 | 0.419 | 0.401 | 0.378 |
| JiT-7B | 0.395 | 0.377 | 0.361 | 0.342 | 0.381 | 0.365 | 0.349 | 0.329 |
| JiT-API | 0.515 | 0.494 | 0.474 | 0.448 | 0.496 | 0.476 | 0.456 | 0.431 |
| FastReasonSeg-8B | 0.678 | 0.650 | 0.624 | 0.599 | 0.659 | 0.630 | 0.605 | 0.579 |
| FastReasonSeg-1.7B-Distill | 0.665 | 0.637 | 0.611 | 0.586 | 0.646 | 0.617 | 0.592 | 0.566 |
| FastReasonSeg-0.6B-Distill | 0.649 | 0.620 | 0.594 | 0.570 | 0.629 | 0.600 | 0.576 | 0.549 |

These are reference values copied from paper Table 2, not reproduced results. Official RVTBench evaluation of this project remains NOT VERIFIED.

## 4. Our local video-sequence results

| Sequence | Frames | IoU | Dice | Seconds/frame | Detections | Errors |
|---|---:|---:|---:|---:|---:|---:|
| Bear | 82 | 96.60% | 98.27% | 1.379 | 82 | 0 |
| Car-turn | 80 | 97.63% | 98.79% | 1.494 | 80 | 0 |
| Bear-evaluation | 5 | N/A | N/A | N/A | 5 | 0 |

### Combined saved-record summary

| Metric | Result |
|---|---:|
| Valid IoU records | 162 |
| Mean IoU | 97.11% |
| Valid Dice records | 162 |
| Mean Dice | 98.53% |
| Mean processing time | 1.436 seconds/frame |

These are summaries of saved local CSV records, not official RVTBench scores.

## 5. Truck single-image mask evaluation

| Prediction mask | IoU (J) | Dice | Boundary F (2 px) |
|---|---:|---:|---:|
| truck_sam2_refined_mask.png | 85.45% | 92.15% | 12.16% |
| truck_refined_mask.png | 95.03% | 97.45% | 25.23% |
| truck_clean_mask.png | 85.74% | 92.32% | 13.32% |
| truck_mask.png | 94.61% | 97.23% | 22.86% |

The refined truck mask has the highest IoU among the four evaluated masks.
This is a separate single-image evaluation, not an RVTBench result.

## 6. Paper versus project comparison

| Paper metric | Our available metric | Comparison status |
|---|---|---|
| RVTBench Region Similarity J | Local DAVIS IoU / truck IoU | Not directly comparable |
| RVTBench Contour Accuracy F | Local truck boundary F | Not directly comparable |
| RVTBench Levels 1–4 | No verified local level results | Not evaluated |
| Paper model efficiency | Local seconds/frame only | Hardware/protocol not matched |

A numerical difference or percentage improvement is not reported because
the datasets, evaluation protocols and test conditions are not established
as equivalent.

## 7. Model configuration

- Base model: Qwen/Qwen3-0.6B
- Adapter: LoRA
- Rank: 8
- Alpha: 16
- Target modules: q_proj, k_proj, v_proj, o_proj
- Adapter weights file: models/final_adapter/adapter_model.safetensors

Adapter presence does not prove full paper training or distillation.

## 8. Limitations and incomplete experiments

- Official RVTBench evaluation has not been verified.
- The previous mask audit found zero usable matched GT/prediction pairs
  for generating new benchmark J/F scores.
- Full paper-specific teacher training and SFT/GRPO/student distillation
  are not implemented according to the existing project report.
- Persistent video tracking has not been independently verified.
- Only a limited number of local sequences are evaluated.
- Saved CSV values require independent annotation and metric validation.

## 9. Reproducibility

```bash
source .venv/bin/activate
python finalize_fastreasonseg.py --root .
python fix_final_report.py
```

## 10. Academic integrity

Paper-reported values and locally measured prototype results are kept
separate. Missing benchmark scores are marked as not evaluated.
No official reproduction claim is made.


## Paper-specific method alignment (from CVPR 2026 paper)

- Teacher: Qwen3-8B. Students: Qwen3-1.7B and Qwen3-0.6B.
- Digital twin inputs are constructed from SAM-2 instance masks, Depth Anything 2 depth statistics, and OWLv2 semantic labels, boxes, and confidence scores.
- Teacher training: GRPO with format reward (+0.5 valid structure, -0.5 invalid) and segmentation accuracy reward (+1 if mask IoU > 0.5, otherwise 0). Required structure includes reason, plan, results, and answer token pairs.
- Student distillation: teacher-generated chains are filtered by reject sampling at predicted IoU > 0.7, then student SFT, followed by RL with format + accuracy + gamma times teacher-reasoning reward. Paper uses gamma=0.5 and GPT-4o as LLM judge (temperature 0.3, max 512 tokens).
- Paper implementation details: LoRA rank 64, alpha 128; eight RTX 4090 GPUs; gradient accumulation 64; effective batch size 512; FP16. Teacher GRPO learning rate 4e-5, 10% warmup and cosine schedule. SFT AdamW learning rate 5e-5, weight decay 0.01, 3 epochs.
- Paper training data: RefCOCOg, ReasonSeg training split, and ReVOS. Paper video benchmarks include JiTBench and RVTBench; image benchmarks include ReasonSeg and LLM-Seg40K.

The project's adapter is Qwen3-0.6B LoRA rank 8 / alpha 16, and its local datasets are mostly motion-calculation examples. This does not match the paper's stated training configuration or task data. A generic SFT/GRPO run on the current files must not be described as FastReasonSeg reproduction.

Paper source: Shen and Unberath, *Fast Reasoning Segmentation for Images and Videos*, CVPR 2026, https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_Fast_Reasoning_Segmentation_for_Images_and_Videos_CVPR_2026_paper.pdf (also arXiv:2511.12368).
