# FastReasonSeg-Inspired Segmentation Prototype
## Final Evaluation Report

Generated: 2026-10-28 07:14

### 1. Project overview

This project implements a FastReasonSeg-inspired prototype combining
object segmentation, temporal mask analysis, a reconstructed Digital Twin,
and image-space motion analysis.

This is a prototype inspired by the research direction, not a complete
reproduction of the original FastReasonSeg training pipeline.

### 2. Video segmentation

- Total frames: 289
- Frames with non-empty segmentation masks: 267
- Frames with empty masks: 22
- Mask availability: 92.39%

A non-empty mask indicates that foreground pixels were returned.
It does not prove that the dog was fully visible or correctly segmented.
An empty mask does not necessarily mean the dog was absent from the scene.

### 3. Motion analysis

- Consecutive visible-frame pairs analyzed: 265
- Mean center displacement: 10.01 pixels/frame
- Median center displacement: 8.47 pixels/frame
- Maximum center displacement: 33.41 pixels
- Largest measured movement: frame 48
- Direction counts: {'left': 106, 'right': 100, 'stationary': 27, 'down': 17, 'up': 15}

These values describe changes in the segmentation-mask center in image
coordinates. They are not physical speed measurements and are not
ground-truth tracking accuracy.

### 4. Visual inspection

Visual inspection of frames 131, 132, and 139 showed that the mask followed
the visible dog reasonably well in frames 132 and 139. Frame 131 showed
partial visibility at the image boundary.

Inspection of frames 104–106 showed the dog entering from the right edge.
The changing mask area is plausibly related to increasing visibility.
These observations are qualitative and do not establish pixel-level accuracy.

### 5. Limitations

- No independently annotated ground-truth masks were used for quantitative
  segmentation accuracy.
- Mask availability and consecutive-frame IoU are not substitutes for
  ground-truth metrics such as mask IoU or Dice.
- Motion is measured from mask centers and may be affected by changes in
  object visibility and mask shape.
- The reconstructed Digital Twin is based on saved segmentation masks;
  missing detection confidence or depth values cannot be recovered from
  those masks alone.
- The complete paper-specific teacher training, SFT/GRPO training, and
  student distillation pipeline was not implemented.

### 6. Recommended future work

1. Inspect remaining suspicious intervals, especially frames 71–78 and
   99–103.
2. Evaluate against independently annotated ground-truth masks.
3. Report ground-truth mask IoU, Dice score, and boundary metrics.
4. Evaluate temporal stability separately from segmentation accuracy.
5. Document all models, checkpoints, settings, and dataset provenance.

### 7. Overall conclusion

The prototype demonstrates a working workflow for video segmentation,
temporal mask storage, Digital Twin reconstruction, and basic motion
analysis. Its current quantitative results describe mask availability and
image-space motion. Further ground-truth-based evaluation is required before
claiming segmentation accuracy or equivalence to the original research paper.



## Latest Prototype Validation Results

### 1. Image Pipeline
- Object detector: OWLv2.
- Image segmenter: SAM2.1 Hiera Small.
- Depth estimator: Depth Anything V2 Small.
- Input image dimensions: 1800 × 1200.
- Refined mask pixels: 639,940.
- SAM2 predicted mask score: 0.97857.
- Depth statistics saved in the Digital Twin JSON.
- Ground-truth image segmentation accuracy: Not measured.

### 2. Mask Comparison
- Baseline mask pixels: 635,969.
- Refined mask pixels: 639,940.
- Baseline/refined mask agreement IoU: 0.9882.
- Baseline/refined mask agreement Dice: 0.9941.
- Note: These agreement scores are not ground-truth accuracy metrics.

### 3. Video Segmentation
- Model: SAM2.1 Hiera Small.
- Device: Apple MPS.
- Input video: docs/recovered_results/videos/01_dog.mp4.
- Output video: data/dog_sam2_segmented.mp4.
- Resolution: 1280 × 720.
- Frame rate: 29.97 FPS.
- Total frames processed: 289.
- Video propagation time: Approximately 7 minutes 31 seconds.
- Exported video size: Approximately 6.68 MB.
- Video export validation: Passed.
- Digital Twin JSON validation: Passed.
- Ground-truth video segmentation accuracy: Not measured.

### 4. Limitations
- Visual inspection covered selected frames, not every frame.
- Ground-truth masks were unavailable for quantitative accuracy evaluation.
- Relative depth values are not physical distances in meters.
- Successful execution of OWLv2, SAM2, and Depth Anything V2 does not establish a complete reproduction of FastReasonSeg.
- Reproduction status requires comparison with the original paper's architecture, implementation, datasets, and evaluation protocol.


## Manual Ground-Truth Evaluation

### Truck Image Segmentation

- Image: `data/truck.jpg`
- Segmentation model: SAM2.1 Hiera Small
- Reference: Manually drawn polygon, converted to a binary mask
- Evaluation scope: One image

| Metric | Result |
|---|---:|
| Ground-truth foreground pixels | 644,003 |
| Predicted foreground pixels | 639,940 |
| Intersection pixels | 625,606 |
| Intersection over Union (IoU) | 95.03% |
| Dice coefficient | 97.45% |

Results are agreement measurements between the SAM2 prediction and
the approximate manual annotation. They are not official benchmark
results and should not be generalized to other images.

The full FastReasonSeg training, reasoning, and distillation pipeline
has not been reproduced. The project is a reduced prototype.


## Additional Features Implemented

### Feature 1: Interactive Text-Based Object Selection

Implemented `interactive_object_query()` to retrieve an object from
the structured Digital Twin using a text query.

Test performed:
- Query: "Find the truck"
- Result: Successfully retrieved the truck object.
- Retrieved information: Object label, bounding box, mask path,
  detector confidence, and mean relative-depth value.

Limitation:
This implementation matches text against existing Digital Twin labels.
It does not perform LLM-based reasoning or run object detection again.

### Feature 2: Spatial Relationship Analysis

Implemented `analyze_scene_relationships()` to calculate object positions,
bounding-box areas, and relationships between objects within the same frame.

Tests performed:
- Existing Digital Twin: Successfully processed one truck.
- Synthetic two-object test: Confirmed left/right position comparison
  and relative-depth value comparison.

Limitation:
The two-object test used synthetic data. Multi-object relationships
have not yet been validated on a real image containing multiple detected
objects. Relative-depth values are not physical distances in meters.

### Current Scope

These additions extend the existing FastReasonSeg-inspired prototype.
They do not constitute a complete reproduction of the original research
method. Full LLM reasoning, GRPO training, student-model distillation,
and official benchmark evaluation remain unimplemented.


## User Input and Prompt Validation

### Features Implemented

1. **Input validation**
   - Rejects empty and whitespace-only inputs.
   - Rejects non-string inputs.
   - Limits prompts to 500 characters.
   - Rejects inputs without letters or numbers.

2. **Interactive object-query validation**
   - Matches queries against existing Digital Twin object labels.
   - Rejects unknown objects and reports available labels.
   - Handles missing or invalid Digital Twin JSON files.

3. **Query-processing validation**
   - Rejects unsupported queries.
   - Returns controlled error responses when query processing fails.

### Validation Test Results

| Test | Expected Behavior | Result |
|---|---|---|
| Where is the truck? | Return object position | PASS |
| Empty query | Reject input | PASS |
| Numeric input (123) | Reject input | PASS |
| Find the airplane | Reject unsupported query | PASS |
| Unknown object in interactive query | Report available objects | PASS |
| Generic request: Find object | Select the only available object | PASS |

### Limitations

- The query planner currently supports truck-related queries only.
- Object selection matches existing Digital Twin labels; it does not detect new objects.
- Input validation does not implement an LLM-based reasoning system.
- These tests verify the prototype's behavior, not the full FastReasonSeg research pipeline.
