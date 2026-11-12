"""Adapter layer connecting the existing app functions to Digital Twin JSON.

Call from app.py after importing this module. The callbacks are injected so this
module does not import Streamlit app.py or duplicate model loading.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
from PIL import Image
from .digital_twin import build_digital_twin


def run_image_digital_twin(
    image: Image.Image,
    prompt: str,
    threshold: float,
    detect_fn: Callable,
    segment_fn: Callable,
    depth_fn: Callable,
) -> tuple[dict, dict[int, np.ndarray]]:
    """Run OWLv2 -> SAM2 -> Depth Anything using existing app callbacks.

    Current app's detect_fn returns (boxes, scores), so labels fall back to prompt.
    A future detector may return (boxes, scores, labels) without changing this API.
    depth_fn is expected to return (PIL depth visualization, statistics), as in app.py.
    """
    image = image.convert("RGB")
    result = detect_fn(image, prompt, threshold)
    if len(result) == 3:
        boxes, scores, labels = result
    else:
        boxes, scores = result
        labels = [prompt] * len(boxes)
    depth_image, _depth_stats = depth_fn(image)
    depth_map = np.asarray(depth_image.convert("L"), dtype=np.float32) / 255.0

    detections, masks = [], {}
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        mask, seg_score = segment_fn(image, box)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (image.height, image.width):
            raise ValueError(f"SAM2 mask {i} shape {mask.shape} does not match image")
        masks[i] = mask
        detections.append({
            "object_id": f"obj_{i+1:03d}",
            "label": str(label),
            "confidence": float(score),
            "box": np.asarray(box, dtype=float).tolist(),
            "mask": mask,
            "segmentation_score": float(seg_score),
        })
    twin = build_digital_twin(
        image_size=(image.width, image.height), detections=detections,
        depth_map=depth_map, source="single_image", prompt=prompt,
    )
    return twin, masks
