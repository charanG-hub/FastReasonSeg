"""Object-level digital-twin representation for FastReasonSeg.

This module deliberately does not claim metric/physical depth. Depth Anything V2
provides relative depth; values below are descriptive statistics over its output.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence
import numpy as np

SCHEMA_VERSION = "1.0"


def _as_box(box: Sequence[float], width: int, height: int) -> list[float]:
    if len(box) != 4:
        raise ValueError("Each box must contain x1, y1, x2, y2")
    x1, y1, x2, y2 = map(float, box)
    x1 = float(np.clip(x1, 0, width)); x2 = float(np.clip(x2, 0, width))
    y1 = float(np.clip(y1, 0, height)); y2 = float(np.clip(y2, 0, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid/empty box after clipping: {[x1,y1,x2,y2]}")
    return [x1, y1, x2, y2]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def object_depth_statistics(depth_map: np.ndarray | None, mask: np.ndarray) -> dict[str, Any] | None:
    """Summarize relative depth-map values within an object mask."""
    if depth_map is None:
        return None
    depth = np.asarray(depth_map, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if depth.ndim != 2 or depth.shape != m.shape:
        raise ValueError("depth_map and mask must be 2-D arrays with identical shapes")
    values = depth[m]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "kind": "relative_depth_model_output",
        "units": "arbitrary_model_units",
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "valid_pixel_count": int(values.size),
        "interpretation_warning": "Relative model output only; not metric distance in metres.",
    }


def build_object_record(
    object_id: str,
    label: str,
    confidence: float,
    box: Sequence[float],
    mask: np.ndarray,
    depth_map: np.ndarray | None = None,
    segmentation_score: float | None = None,
) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    h, w = m.shape
    b = _as_box(box, w, h)
    area = int(m.sum())
    ys, xs = np.where(m)
    if area:
        centroid = {"x": float(xs.mean()), "y": float(ys.mean())}
        mask_bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    else:
        centroid = None
        mask_bbox = None
    x1, y1, x2, y2 = b
    box_area = max((x2 - x1) * (y2 - y1), 1.0)
    return {
        "object_id": str(object_id),
        "label": str(label or "unknown"),
        "detection_confidence": _safe_float(confidence),
        "segmentation_score": None if segmentation_score is None else _safe_float(segmentation_score),
        "bounding_box_xyxy": [round(v, 3) for v in b],
        "mask": {
            "encoding": "binary_array_not_embedded",
            "height": h,
            "width": w,
            "area_pixels": area,
            "area_fraction_of_image": float(area / (h * w)) if h * w else 0.0,
            "area_fraction_of_detection_box": float(area / box_area),
            "mask_bounding_box_xyxy": mask_bbox,
            "centroid_xy": centroid,
        },
        "relative_depth": object_depth_statistics(depth_map, m),
    }


def build_digital_twin(
    image_size: tuple[int, int],
    detections: Iterable[Mapping[str, Any]],
    depth_map: np.ndarray | None = None,
    source: str = "image",
    prompt: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable scene graph from detection/mask records.

    Each detection must have box and mask; label and confidence are optional.
    image_size is (width, height).
    """
    width, height = map(int, image_size)
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")
    objects = []
    for index, det in enumerate(detections):
        mask = np.asarray(det["mask"], dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(f"Detection {index} mask shape {mask.shape} != {(height, width)}")
        objects.append(build_object_record(
            object_id=det.get("object_id", f"obj_{index + 1:03d}"),
            label=det.get("label", prompt or "unknown"),
            confidence=det.get("confidence", 0.0),
            box=det["box"],
            mask=mask,
            depth_map=depth_map,
            segmentation_score=det.get("segmentation_score"),
        ))
    # Compute 3D Spatial-Geometric Graph and Relational Grounding
    from .spatial_depth_gate import compute_3d_spatial_graph
    spatial_graph = compute_3d_spatial_graph(objects, width, height)

    return {
        "schema": "fastreasonseg.digital_twin",
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "prompt": prompt,
        "image": {"width": width, "height": height},
        "object_count": len(objects),
        "objects": objects,
        "spatial_3d_graph": spatial_graph,
        "limitations": [
            "Masks are summarized, not embedded; save mask arrays separately if needed.",
            "Depth values are relative model outputs, not calibrated physical distances.",
            "Object labels inherit the detector prompt unless explicit labels are supplied.",
        ],
    }


def save_digital_twin(twin: Mapping[str, Any], path: str) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(twin, f, indent=2, ensure_ascii=False)
