# Combined Segmentation Benchmark Report

Generated: 2026-10-30T11:33:32+05:30

> These are prototype results on DAVIS sequences. They are not official RVTBench results and do not establish a full reproduction of the FastReasonSeg paper.

## Per-sequence results

| Sequence | Frames | Detections | Mean IoU | Mean Dice | Mean seconds/frame | Errors |
|---|---:|---:|---:|---:|---:|---:|
| Bear | 82 | 82 | 0.966016 | 0.982706 | 1.379 | 0 |
| Car-turn | 80 | 80 | 0.976261 | 0.987947 | 1.494 | 0 |

## Combined results

- Total frame records: 162
- Rows with IoU values: 162
- Rows with Dice values: 162
- Rows with timing values: 162
- Combined mean IoU: 0.971075
- Combined mean Dice: 0.985294
- Combined mean processing time: 1.436 seconds/frame

## Interpretation and limitations

- IoU and Dice values are calculated from the saved per-frame CSV records.
- Combined metrics use individual frame values, rather than averaging sequence means.
- File counts and CSV values alone do not independently validate mask alignment or metric implementation.
- These runs use independent frame-level detection and segmentation; they should not be described as persistent video tracking unless that capability is separately verified.
- Two sequences are a limited evaluation and are not sufficient to establish generalization or state-of-the-art performance.
- Verify the evaluation script, annotation alignment, and mask thresholding before treating these numbers as final scientific results.
