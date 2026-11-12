# DAVIS Bear Segmentation Benchmark Report

**Generated:** 2026-10-30

## 1. Objective

Evaluate the existing text-guided detection and segmentation prototype on
the DAVIS `bear` sequence using its frame-level ground-truth masks.

## 2. Method

- Sequence: DAVIS `bear`
- Input frames: 82
- Prompt: `bear`
- Detection: OWLv2 text-guided object detection
- Segmentation: SAM 2.1 Small through the existing prototype
- Frame processing: each frame is detected and segmented independently
- Metrics: Intersection over Union (IoU) and Dice coefficient
- Ground truth: DAVIS binary annotation masks

## 3. Results

| Metric | Result |
|---|---:|
| Frames in CSV | 82 |
| Frames with valid metrics | 82 |
| Frames with errors or missing metrics | 0 |
| Mean IoU | 0.966016 |
| Median IoU | 0.966693 |
| Minimum IoU | 0.945021 |
| Maximum IoU | 0.976402 |
| Mean Dice | 0.982706 |
| Median Dice | 0.983065 |
| Minimum Dice | 0.971734 |
| Maximum Dice | 0.988060 |
| Mean measured processing time/frame | 1.379354 seconds |
| Median measured processing time/frame | 1.282500 seconds |

The mean processing time includes first-frame initialization overhead if it
was recorded in the original run. It should not be interpreted as steady-state
throughput without a separate timing analysis.

## 4. Verification

The saved predicted masks were independently compared with the DAVIS ground
truth. All 82 masks were found, no metric mismatches were reported, and the
recalculated means matched the CSV.

## 5. Limitations

1. This is a single-sequence benchmark; results do not establish performance
   across diverse objects or videos.
2. The prototype performs independent frame-level detection and segmentation;
   it does not establish persistent video tracking.
3. The results are not official RVTBench metrics.
4. This experiment alone does not constitute a full reproduction of the
   FastReasonSeg research paper.
5. Performance on other sequences, prompts, and challenging scenes remains
   to be evaluated.

## 6. Artifacts

- Raw frame-level results: `data/bear_benchmark/results.csv`
- Summary CSV: `data/bear_benchmark_summary.csv`
- Per-frame metric graph: `data/bear_benchmark/bear_iou_dice_graph.png`

## 7. Conclusion

The existing DAVIS Bear run has internally consistent frame-level metrics,
with mean IoU 0.966016 and mean Dice 0.982706.
These results support a preliminary single-sequence evaluation of the
prototype. Broader evaluation is needed before making general performance
claims.
