"""Lightweight box-association tracker for stable IDs across video frames.

This is NOT SAM2 video propagation. It associates per-frame detections using
IoU + center distance and is intended as a practical baseline/fallback.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a); bx1, by1, bx2, by2 = map(float, b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: list[float]
    label: str
    confidence: float
    last_frame: int
    age: int = 0
    hits: int = 1


@dataclass
class IoUTracker:
    iou_threshold: float = 0.15
    max_age: int = 8
    center_distance_threshold: float = 0.20  # normalized by image diagonal
    tracks: dict[int, Track] = field(default_factory=dict)
    next_id: int = 1

    def update(self, detections: list[dict[str, Any]], frame_index: int,
               image_size: tuple[int, int]) -> list[dict[str, Any]]:
        width, height = image_size
        diag = max(float(np.hypot(width, height)), 1.0)
        candidates = []
        for tid, track in self.tracks.items():
            for di, det in enumerate(detections):
                if track.label != det.get("label", track.label):
                    continue
                iou = box_iou(track.box, det["box"])
                a = np.array([(track.box[0]+track.box[2])/2, (track.box[1]+track.box[3])/2])
                b = np.array([(det["box"][0]+det["box"][2])/2, (det["box"][1]+det["box"][3])/2])
                dist = float(np.linalg.norm(a-b) / diag)
                if iou >= self.iou_threshold or dist <= self.center_distance_threshold:
                    # Higher IoU and smaller normalized center distance are preferred.
                    candidates.append((iou - dist, tid, di))
        candidates.sort(reverse=True)
        used_tracks, used_dets, assignments = set(), set(), {}
        for _, tid, di in candidates:
            if tid not in used_tracks and di not in used_dets:
                assignments[di] = tid; used_tracks.add(tid); used_dets.add(di)

        output = []
        for di, det in enumerate(detections):
            if di in assignments:
                tid = assignments[di]
                track = self.tracks[tid]
                track.box = list(map(float, det["box"]))
                track.confidence = float(det.get("confidence", 0.0))
                track.last_frame = int(frame_index); track.age = 0; track.hits += 1
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = Track(tid, list(map(float, det["box"])),
                    str(det.get("label", "unknown")), float(det.get("confidence", 0.0)), int(frame_index))
            row = dict(det); row["track_id"] = f"track_{tid:04d}"
            output.append(row)

        for tid, track in list(self.tracks.items()):
            if tid not in used_tracks and tid not in assignments.values():
                track.age = int(frame_index) - track.last_frame
            if track.age > self.max_age:
                del self.tracks[tid]
        return output
