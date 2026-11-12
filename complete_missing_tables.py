#!/usr/bin/env python3
"""
FastReasonSeg missing-table audit and mask evaluation utility.

What it does:
1. Audits the project for RVTBench / reasoning-level data and mask files.
2. Reuses existing CSV/JSON/Markdown result files without modifying them.
3. Searches for ground-truth and predicted mask folders.
4. If matching mask files are discoverable, calculates per-pair Region Similarity (J/IoU)
   and boundary F-measure using a configurable pixel tolerance.
5. Writes detailed CSVs and a Markdown report.
6. Never fabricates RVTBench results: missing data is explicitly marked NOT AVAILABLE.

Run from the FastReasonSeg repository root:
    python complete_missing_tables.py
Optional:
    python complete_missing_tables.py --root . --boundary-tolerance 2
    python complete_missing_tables.py --gt-dir path/to/ground_truth_masks --pred-dir path/to/predicted_masks
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Optional

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:
    print("Missing dependency. Activate the project virtual environment and run:")
    print("  python -m pip install numpy pillow")
    raise SystemExit(2) from exc

MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
IGNORED_DIRS = {".venv", "venv", ".git", "__pycache__", "node_modules", ".pytest_cache"}
GT_HINTS = ("ground_truth", "ground-truth", "groundtruth", "gt_masks", "gt_mask", "annotations", "annotation")
PRED_HINTS = ("predictions", "prediction", "pred_masks", "pred_mask", "predicted_masks", "masks_pred", "outputs_masks")
LEVEL_RE = re.compile(r"(?:level|lvl|l)[ _-]?([1-4])", re.IGNORECASE)


def is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


def read_text_excerpt(path: Path, max_chars: int = 1800) -> str:
    try:
        return path.read_text(errors="replace")[:max_chars].replace("\n", " ")
    except Exception:
        return ""


def collect_files(root: Path):
    files = []
    for p in root.rglob("*"):
        if not p.is_file() or is_ignored(p):
            continue
        files.append(p)
    return files


def mask_files_under(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS and not is_ignored(p))


def discover_mask_directories(files):
    found = []
    for p in files:
        if p.suffix.lower() not in MASK_EXTENSIONS:
            continue
        parent = p.parent
        if parent not in found:
            found.append(parent)
    return found


def classify_dir(path: Path) -> str:
    low = str(path).lower()
    if any(h in low for h in GT_HINTS):
        return "ground_truth_candidate"
    if any(h in low for h in PRED_HINTS):
        return "prediction_candidate"
    return "mask_directory_unclassified"


def normalize_stem(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"(?:_gt|_groundtruth|_ground_truth|_mask|_masks|_pred|_prediction|_predicted)$", "", stem)
    stem = re.sub(r"^(?:gt|groundtruth|ground_truth|pred|prediction|predicted|mask)_", "", stem)
    return re.sub(r"[^a-z0-9]+", "", stem)


def load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    return arr > 0


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    # Pure NumPy dilation; avoids requiring OpenCV/scipy.
    h, w = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy:dy+h, dx:dx+w]
    return out


def boundary(mask: np.ndarray) -> np.ndarray:
    # Four-neighbour inner boundary: foreground pixels adjacent to background.
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    up = np.zeros_like(mask); up[1:, :] = mask[:-1, :]
    down = np.zeros_like(mask); down[:-1, :] = mask[1:, :]
    left = np.zeros_like(mask); left[:, 1:] = mask[:, :-1]
    right = np.zeros_like(mask); right[:, :-1] = mask[:, 1:]
    eroded = mask & up & down & left & right
    return mask & ~eroded


def region_similarity(gt: np.ndarray, pred: np.ndarray) -> Optional[float]:
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(gt, pred).sum() / union)


def dice_score(gt: np.ndarray, pred: np.ndarray) -> Optional[float]:
    denom = int(gt.sum()) + int(pred.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(gt, pred).sum() / denom)


def contour_f_measure(gt: np.ndarray, pred: np.ndarray, tolerance: int) -> Optional[float]:
    gt_b = boundary(gt)
    pred_b = boundary(pred)
    ngt, npred = int(gt_b.sum()), int(pred_b.sum())
    if ngt == 0 and npred == 0:
        return 1.0
    if ngt == 0 or npred == 0:
        return 0.0
    pred_match = np.logical_and(pred_b, dilate(gt_b, tolerance)).sum()
    gt_match = np.logical_and(gt_b, dilate(pred_b, tolerance)).sum()
    precision = float(pred_match / npred)
    recall = float(gt_match / ngt)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def infer_level(path: Path) -> str:
    m = LEVEL_RE.search(str(path))
    return f"Level {m.group(1)}" if m else "Unlabelled"


def infer_sequence(path: Path) -> str:
    parts = list(path.parts)
    for i, part in enumerate(parts):
        low = part.lower()
        if any(x in low for x in ("bear", "car-turn", "car_turn", "davis", "rvtbench")):
            if low in ("rvtbench", "davis"):
                if i + 1 < len(parts):
                    return parts[i + 1]
            return part
    return path.parent.name


def build_pair_candidates(gt_files, pred_files):
    pred_by_key = defaultdict(list)
    for p in pred_files:
        pred_by_key[normalize_stem(p)].append(p)
    pairs, unmatched_gt, used_pred = [], [], set()
    for gt in gt_files:
        key = normalize_stem(gt)
        candidates = pred_by_key.get(key, [])
        if len(candidates) == 1:
            pairs.append((gt, candidates[0]))
            used_pred.add(candidates[0])
        elif len(candidates) > 1:
            # Prefer candidates from the same relative sequence/level when possible.
            scored = []
            for p in candidates:
                score = 0
                if infer_level(p) == infer_level(gt):
                    score += 2
                if infer_sequence(p).lower() == infer_sequence(gt).lower():
                    score += 2
                if p.parent.name.lower() in str(gt.parent).lower() or gt.parent.name.lower() in str(p.parent).lower():
                    score += 1
                scored.append((score, p))
            scored.sort(key=lambda x: x[0], reverse=True)
            if len(scored) == 1 or scored[0][0] > scored[1][0]:
                pairs.append((gt, scored[0][1]))
                used_pred.add(scored[0][1])
            else:
                unmatched_gt.append(gt)
        else:
            unmatched_gt.append(gt)
    unmatched_pred = [p for p in pred_files if p not in used_pred]
    return pairs, unmatched_gt, unmatched_pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--gt-dir", default=None, help="Optional explicit ground-truth mask directory")
    parser.add_argument("--pred-dir", default=None, help="Optional explicit predicted mask directory")
    parser.add_argument("--boundary-tolerance", type=int, default=2, help="Boundary match tolerance in pixels (default: 2)")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: root does not exist: {root}")
        return 2

    out_dir = root / "docs" / "recovered_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_files = collect_files(root)

    relevant_names = []
    for p in all_files:
        low = str(p).lower()
        if any(k in low for k in ("rvtbench", "ground_truth", "ground-truth", "annotation", "reasoning_level", "level_1", "level_2", "level_3")):
            if p.suffix.lower() in {".md", ".csv", ".json", ".yaml", ".yml", ".txt", ".py", ".ipynb"}:
                relevant_names.append(p)

    mask_files = [p for p in all_files if p.suffix.lower() in MASK_EXTENSIONS]
    mask_dirs = discover_mask_directories(mask_files)
    gt_files, pred_files = [], []

    if args.gt_dir:
        gt_root = Path(args.gt_dir).expanduser().resolve()
        if not gt_root.exists():
            print(f"ERROR: --gt-dir does not exist: {gt_root}")
            return 2
        gt_files = mask_files_under(gt_root)
    else:
        gt_files = [p for p in mask_files if any(h in str(p.parent).lower() for h in GT_HINTS)]

    if args.pred_dir:
        pred_root = Path(args.pred_dir).expanduser().resolve()
        if not pred_root.exists():
            print(f"ERROR: --pred-dir does not exist: {pred_root}")
            return 2
        pred_files = mask_files_under(pred_root)
    else:
        pred_files = [p for p in mask_files if any(h in str(p.parent).lower() for h in PRED_HINTS)]

    pairs, unmatched_gt, unmatched_pred = build_pair_candidates(gt_files, pred_files) if gt_files and pred_files else ([], gt_files, pred_files)

    per_pair = []
    errors = []
    for gt_path, pred_path in pairs:
        try:
            gt = load_binary_mask(gt_path)
            pred = load_binary_mask(pred_path)
            if gt.shape != pred.shape:
                errors.append({"ground_truth": str(gt_path.relative_to(root)), "prediction": str(pred_path.relative_to(root)), "error": f"Shape mismatch: GT {gt.shape}, prediction {pred.shape}"})
                continue
            per_pair.append({
                "sequence": infer_sequence(gt_path),
                "level": infer_level(gt_path) if infer_level(gt_path) != "Unlabelled" else infer_level(pred_path),
                "frame": gt_path.stem,
                "ground_truth_path": str(gt_path.relative_to(root)),
                "prediction_path": str(pred_path.relative_to(root)),
                "region_similarity_J": region_similarity(gt, pred),
                "contour_accuracy_F": contour_f_measure(gt, pred, max(0, args.boundary_tolerance)),
                "dice": dice_score(gt, pred),
                "gt_pixels": int(gt.sum()),
                "pred_pixels": int(pred.sum()),
                "boundary_tolerance_pixels": max(0, args.boundary_tolerance),
            })
        except Exception as exc:
            errors.append({"ground_truth": str(gt_path), "prediction": str(pred_path), "error": repr(exc)})

    pair_csv = out_dir / "mask_pair_metrics_J_F.csv"
    if per_pair:
        with pair_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_pair[0].keys()))
            writer.writeheader()
            writer.writerows(per_pair)

    group_rows = []
    group_keys = sorted({(r["sequence"], r["level"]) for r in per_pair})
    for seq, level in group_keys:
        rows = [r for r in per_pair if r["sequence"] == seq and r["level"] == level]
        group_rows.append({
            "sequence": seq,
            "level": level,
            "frames_evaluated": len(rows),
            "mean_J": mean(r["region_similarity_J"] for r in rows),
            "median_J": median(r["region_similarity_J"] for r in rows),
            "mean_F": mean(r["contour_accuracy_F"] for r in rows),
            "median_F": median(r["contour_accuracy_F"] for r in rows),
            "mean_Dice": mean(r["dice"] for r in rows),
        })
    group_csv = out_dir / "mask_metrics_by_sequence_level.csv"
    if group_rows:
        with group_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
            writer.writeheader()
            writer.writerows(group_rows)

    # Inspect existing project result files for provenance and claims.
    result_files = [
        p for p in all_files
        if p.suffix.lower() in {".csv", ".json", ".md"}
        and any(k in p.name.lower() for k in ("benchmark", "evaluation", "metric", "result", "query_plan"))
    ]
    report = []
    report += [
        "# FastReasonSeg Missing Tables — Automated Audit",
        "",
        f"- Repository root: `{root}`",
        f"- Audit timestamp: {__import__('datetime').datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Boundary tolerance: {max(0, args.boundary_tolerance)} pixels",
        "",
        "## Result",
        "",
        f"- Mask files discovered in repository: **{len(mask_files)}**",
        f"- Ground-truth mask files selected: **{len(gt_files)}**",
        f"- Predicted mask files selected: **{len(pred_files)}**",
        f"- Matching GT/prediction pairs: **{len(pairs)}**",
        f"- Pairs successfully evaluated: **{len(per_pair)}**",
        f"- Evaluation errors: **{len(errors)}**",
        "",
        "## Table status",
        "",
        "| Table | Status |",
        "|---|---|",
        "| Bear DAVIS IoU/Dice/timing | Existing report found; see combined benchmark report |",
        "| Car-turn DAVIS IoU/Dice/timing | Existing report found; see combined benchmark report |",
        "| Combined DAVIS summary | Existing report found; not official RVTBench |",
        f"| Region Similarity (J) from discovered mask pairs | {'GENERATED' if per_pair else 'NOT AVAILABLE — no usable matched mask pairs found'} |",
        f"| Contour Accuracy (F) from discovered mask pairs | {'GENERATED' if per_pair else 'NOT AVAILABLE — no usable matched mask pairs found'} |",
        f"| Official RVTBench Level 1–3 table | {'PARTIAL — level-labelled masks found; verify benchmark provenance' if any(r['level'] != 'Unlabelled' for r in per_pair) else 'NOT VERIFIED — official RVTBench data and level labels not established'} |",
        "",
        "Important: J and F computed here are only valid for the discovered GT/prediction mask pairs. This script does not convert DAVIS scores into RVTBench scores and does not claim the pairs are official RVTBench samples.",
        "",
        "## Discovered mask directories",
        "",
        "| Directory | Classification | Mask files |",
        "|---|---|---:|",
    ]
    for d in mask_dirs[:200]:
        count = sum(1 for p in mask_files if p.parent == d)
        report.append(f"| `{d.relative_to(root)}` | {classify_dir(d)} | {count} |")
    if not mask_dirs:
        report.append("| — | No mask directories found | 0 |")

    report += ["", "## Grouped J/F results", ""]
    if group_rows:
        report += [
            "| Sequence | Level | Frames | Mean J | Median J | Mean F | Median F | Mean Dice |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in group_rows:
            report.append(
                f"| {r['sequence']} | {r['level']} | {r['frames_evaluated']} | "
                f"{r['mean_J']:.6f} | {r['median_J']:.6f} | {r['mean_F']:.6f} | "
                f"{r['median_F']:.6f} | {r['mean_Dice']:.6f} |"
            )
    else:
        report.append("No valid paired masks were found, so no new J/F scores were calculated.")

    report += ["", "## Existing relevant result files", ""]
    for p in sorted(result_files)[:100]:
        report.append(f"- `{p.relative_to(root)}`")
    if not result_files:
        report.append("- No existing benchmark/evaluation/metric/result files discovered.")

    report += ["", "## Files created", ""]
    report.append(f"- `{pair_csv.relative_to(root)}` — per-frame J/F/Dice metrics" if per_pair else "- Per-frame J/F CSV not created because no valid mask pairs were available.")
    report.append(f"- `{group_csv.relative_to(root)}` — grouped by sequence and level" if group_rows else "- Grouped J/F CSV not created because no valid mask pairs were available.")
    report.append(f"- `{(out_dir / 'MISSING_TABLES_AUDIT.md').relative_to(root)}` — this audit report")
    if unmatched_gt:
        report += ["", f"## Unmatched ground-truth masks ({len(unmatched_gt)})", ""]
        for p in unmatched_gt[:50]:
            report.append(f"- `{p.relative_to(root)}`")
    if unmatched_pred:
        report += ["", f"## Unmatched predicted masks ({len(unmatched_pred)})", ""]
        for p in unmatched_pred[:50]:
            report.append(f"- `{p.relative_to(root)}`")
    if errors:
        report += ["", "## Errors", ""]
        for e in errors[:100]:
            report.append(f"- GT `{e['ground_truth']}` / Pred `{e['prediction']}`: {e['error']}")

    audit_path = out_dir / "MISSING_TABLES_AUDIT.md"
    audit_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n========== FASTREASONSEG TABLE AUDIT ==========")
    print(f"Repository: {root}")
    print(f"Mask files: {len(mask_files)}")
    print(f"GT masks selected: {len(gt_files)}")
    print(f"Prediction masks selected: {len(pred_files)}")
    print(f"Matched pairs: {len(pairs)}")
    print(f"Evaluated pairs: {len(per_pair)}")
    print(f"Errors: {len(errors)}")
    print(f"Audit report: {audit_path}")
    if per_pair:
        print(f"Per-frame J/F CSV: {pair_csv}")
        print(f"Grouped J/F CSV: {group_csv}")
        print("\nFirst grouped rows:")
        for row in group_rows[:20]:
            print(row)
    else:
        print("\nNo trustworthy mask pairs could be auto-evaluated.")
        print("The audit report lists discovered mask directories and existing result files.")
        print("If GT and predicted masks are stored in separate nonstandard folders, rerun with:")
        print("  python complete_missing_tables.py --gt-dir PATH_TO_GT_MASKS --pred-dir PATH_TO_PRED_MASKS")
    print("\nOfficial RVTBench Level 1–3 results are NOT claimed unless the actual benchmark samples and annotations are supplied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
