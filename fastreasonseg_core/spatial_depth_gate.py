"""SpatialDepthGate: 3D Depth-Aware Spatial-Geometric Cross-Gating Module for FastReasonSeg.

Enhances object-level digital twins with explicit 3D relative positioning, depth ranking,
occlusion graphs, and spatial relational grounding. Resolves ambiguities in referring
segmentation (e.g. 'the car behind the truck', 'the person closest to camera').
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def normalize_bbox(box: List[float], width: int, height: int) -> Tuple[float, float, float, float, float, float]:
    """Normalize box coordinates to [0, 1] and compute normalized center and dimensions."""
    x1, y1, x2, y2 = box
    w = max(width, 1)
    h = max(height, 1)
    nx1 = float(np.clip(x1 / w, 0.0, 1.0))
    ny1 = float(np.clip(y1 / h, 0.0, 1.0))
    nx2 = float(np.clip(x2 / w, 0.0, 1.0))
    ny2 = float(np.clip(y2 / h, 0.0, 1.0))
    cx = (nx1 + nx2) / 2.0
    cy = (ny1 + ny2) / 2.0
    nw = max(nx2 - nx1, 1e-4)
    nh = max(ny2 - ny1, 1e-4)
    return cx, cy, nw, nh, nx1, ny1


def compute_box_iou(b1: List[float], b2: List[float]) -> float:
    """Compute 2D Intersection over Union between two xyxy boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


def compute_3d_spatial_graph(
    objects: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    depth_epsilon: float = 0.05,
    spatial_epsilon: float = 0.05,
) -> Dict[str, Any]:
    """Build pairwise 3D spatial relations, relative depth rankings, and occlusion graph.
    
    Args:
        objects: List of object records from digital twin.
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.
        depth_epsilon: Threshold for relative depth equivalence (coplanarity).
        spatial_epsilon: Threshold for horizontal/vertical alignment.
        
    Returns:
        Structured dictionary containing relative 3D coordinates, depth ordering,
        pairwise relation matrix, and natural-language relational summaries.
    """
    n = len(objects)
    if n == 0:
        return {
            "num_objects": 0,
            "depth_ordering": [],
            "spatial_coordinates_3d": [],
            "pairwise_relations": [],
            "relational_prompt_summary": "No objects present in scene.",
        }

    # 1. Extract 3D normalized coordinates
    depth_values = []
    for obj in objects:
        rd = obj.get("relative_depth")
        if rd and isinstance(rd, dict) and "mean" in rd:
            depth_values.append(float(rd["mean"]))
        else:
            depth_values.append(0.5)

    d_min = min(depth_values) if depth_values else 0.0
    d_max = max(depth_values) if depth_values else 1.0
    d_range = max(d_max - d_min, 1e-5)

    coords_3d = []
    for i, obj in enumerate(objects):
        box = obj.get("bounding_box_xyxy", [0, 0, image_width, image_height])
        cx, cy, nw, nh, nx1, ny1 = normalize_bbox(box, image_width, image_height)
        # Normalize depth to [0, 1] where 0 is closest (foreground) and 1 is farthest (background)
        # Depth Anything V2 outputs higher values for closer objects or vice versa depending on inversion;
        # Here we normalize linearly to relative depth rank
        raw_depth = depth_values[i]
        norm_depth = float((raw_depth - d_min) / d_range)
        
        coords_3d.append({
            "object_id": obj.get("object_id", str(i)),
            "label": obj.get("label", "unknown"),
            "x_norm": round(cx, 4),
            "y_norm": round(cy, 4),
            "z_rel": round(norm_depth, 4),
            "width_norm": round(nw, 4),
            "height_norm": round(nh, 4),
            "box_xyxy": box,
        })

    # 2. Compute Depth Order (sorted by z_rel ascending = closest to farthest)
    sorted_by_depth = sorted(
        range(n),
        key=lambda idx: coords_3d[idx]["z_rel"]
    )
    depth_ordering = [
        {
            "depth_rank": rank + 1,
            "object_id": coords_3d[idx]["object_id"],
            "label": coords_3d[idx]["label"],
            "z_rel": coords_3d[idx]["z_rel"],
        }
        for rank, idx in enumerate(sorted_by_depth)
    ]

    # 3. Compute Pairwise Relations Matrix
    pairwise_relations = []
    summary_sentences = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ci = coords_3d[i]
            cj = coords_3d[j]
            dx = ci["x_norm"] - cj["x_norm"]
            dy = ci["y_norm"] - cj["y_norm"]
            dz = ci["z_rel"] - cj["z_rel"]

            # Horizontal relation
            if dx < -spatial_epsilon:
                h_rel = "left_of"
            elif dx > spatial_epsilon:
                h_rel = "right_of"
            else:
                h_rel = "horizontally_aligned_with"

            # Vertical relation
            if dy < -spatial_epsilon:
                v_rel = "above"
            elif dy > spatial_epsilon:
                v_rel = "below"
            else:
                v_rel = "vertically_aligned_with"

            # Depth relation
            if dz < -depth_epsilon:
                d_rel = "foreground_of"
            elif dz > depth_epsilon:
                d_rel = "background_of"
            else:
                d_rel = "coplanar_with"

            # 3D relative distance
            dist_3d = math.sqrt(dx * dx + dy * dy + (dz * 1.5) ** 2)

            # Occlusion check: 2D bounding boxes overlap significantly
            iou_2d = compute_box_iou(ci["box_xyxy"], cj["box_xyxy"])
            is_occluding = bool(iou_2d > 0.08 and dz < -depth_epsilon)

            relation_entry = {
                "source_id": ci["object_id"],
                "target_id": cj["object_id"],
                "source_label": ci["label"],
                "target_label": cj["label"],
                "depth_relation": d_rel,
                "horizontal_relation": h_rel,
                "vertical_relation": v_rel,
                "distance_3d": round(dist_3d, 4),
                "box_iou_2d": round(iou_2d, 4),
                "is_occluding": is_occluding,
            }
            pairwise_relations.append(relation_entry)

            # Natural language description for reasoning prompt augmentation
            if i < j:  # symmetric reduction for prompt length efficiency
                occ_str = " (partially occluding)" if is_occluding else ""
                summary_sentences.append(
                    f"Object {ci['object_id']} ({ci['label']}) is {d_rel} and {h_rel} Object {cj['object_id']} ({cj['label']}){occ_str}."
                )

    return {
        "num_objects": n,
        "spatial_coordinates_3d": coords_3d,
        "depth_ordering": depth_ordering,
        "pairwise_relations": pairwise_relations,
        "relational_prompt_summary": " ".join(summary_sentences) if summary_sentences else "Single object scene.",
    }


class SpatialDepthGate:
    """Neural/Algorithmic Gating head that re-ranks candidate objects based on referring expressions
    and 3D geometric relations.
    """
    def __init__(self, use_neural_head: bool = False, embedding_dim: int = 64):
        self.use_neural_head = use_neural_head
        self.embedding_dim = embedding_dim
        self.weights = {
            "foreground": {"z_weight": -1.5, "keywords": ["front", "closest", "near", "foremost", "ahead", "closer"]},
            "background": {"z_weight": 1.5, "keywords": ["back", "behind", "farthest", "rear", "deep", "distant"]},
            "left": {"x_weight": -1.2, "keywords": ["left", "leftmost", "west"]},
            "right": {"x_weight": 1.2, "keywords": ["right", "rightmost", "east"]},
            "top": {"y_weight": -1.0, "keywords": ["top", "upper", "above", "highest"]},
            "bottom": {"y_weight": 1.0, "keywords": ["bottom", "lower", "below", "lowest"]},
        }

    def score_candidates(
        self,
        query: str,
        spatial_graph: Dict[str, Any],
        initial_confidences: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Score and re-rank candidate objects matching query text with 3D spatial alignment.
        
        Args:
            query: Referring natural language expression (e.g. 'car in the background on the left').
            spatial_graph: Output from compute_3d_spatial_graph.
            initial_confidences: Optional detection confidence scores.
            
        Returns:
            List of candidates with composite scores and rank explanation.
        """
        q_lower = query.lower()
        coords = spatial_graph.get("spatial_coordinates_3d", [])
        n = len(coords)
        if n == 0:
            return []

        if initial_confidences is None:
            initial_confidences = [1.0 / n] * n

        results = []
        for i, c in enumerate(coords):
            base_score = float(initial_confidences[i]) if i < len(initial_confidences) else 0.5
            spatial_bonus = 0.0
            reasons = []

            # Check depth intents
            for kw in self.weights["foreground"]["keywords"]:
                if kw in q_lower:
                    # Prefer lower z_rel (closer)
                    bonus = (1.0 - c["z_rel"]) * 1.5
                    spatial_bonus += bonus
                    reasons.append(f"foreground affinity (+{bonus:.2f})")
                    break

            for kw in self.weights["background"]["keywords"]:
                if kw in q_lower:
                    # Prefer higher z_rel (farther)
                    bonus = c["z_rel"] * 1.5
                    spatial_bonus += bonus
                    reasons.append(f"background affinity (+{bonus:.2f})")
                    break

            # Check horizontal intents
            for kw in self.weights["left"]["keywords"]:
                if kw in q_lower:
                    bonus = (1.0 - c["x_norm"]) * 1.0
                    spatial_bonus += bonus
                    reasons.append(f"left positioning (+{bonus:.2f})")
                    break

            for kw in self.weights["right"]["keywords"]:
                if kw in q_lower:
                    bonus = c["x_norm"] * 1.0
                    spatial_bonus += bonus
                    reasons.append(f"right positioning (+{bonus:.2f})")
                    break

            # Check vertical intents
            for kw in self.weights["top"]["keywords"]:
                if kw in q_lower:
                    bonus = (1.0 - c["y_norm"]) * 0.8
                    spatial_bonus += bonus
                    reasons.append(f"upper positioning (+{bonus:.2f})")
                    break

            for kw in self.weights["bottom"]["keywords"]:
                if kw in q_lower:
                    bonus = c["y_norm"] * 0.8
                    spatial_bonus += bonus
                    reasons.append(f"lower positioning (+{bonus:.2f})")
                    break

            composite_score = base_score + spatial_bonus
            results.append({
                "object_id": c["object_id"],
                "label": c["label"],
                "base_confidence": round(base_score, 4),
                "spatial_bonus": round(spatial_bonus, 4),
                "composite_score": round(composite_score, 4),
                "z_rel": c["z_rel"],
                "x_norm": c["x_norm"],
                "y_norm": c["y_norm"],
                "reasons": reasons,
            })

        # Sort descending by composite score
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        for rank, r in enumerate(results):
            r["rank"] = rank + 1

        return results
