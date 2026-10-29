from pathlib import Path

path = Path("app.py")
source = path.read_text()

old = '''def segment_image(image, box):
    predictor = load_sam()

    rgb = np.asarray(image.convert("RGB"))
    predictor.set_image(rgb)

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            box=np.asarray(box, dtype=np.float32),
            multimask_output=True,
        )

    best_index = int(np.argmax(scores))

    mask = np.asarray(masks[best_index]).astype(bool)
    score = float(scores[best_index])

    return mask, score
'''

new = '''def segment_image(image, box, point_coords=None, point_labels=None):
    """Segment an object with a bounding box and optional point prompts."""
    predictor = load_sam()

    rgb = np.asarray(image.convert("RGB"))
    predictor.set_image(rgb)

    predict_args = {
        "box": np.asarray(box, dtype=np.float32),
        "multimask_output": True,
    }

    if point_coords is not None and len(point_coords) > 0:
        coords = np.asarray(point_coords, dtype=np.float32)
        labels = np.asarray(point_labels, dtype=np.int32)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Point coordinates must have shape (N, 2).")

        if len(coords) != len(labels):
            raise ValueError("Each point must have a corresponding label.")

        if not np.all(np.isfinite(coords)):
            raise ValueError("Point coordinates must be finite.")

        if (
            np.any(coords[:, 0] < 0)
            or np.any(coords[:, 0] >= image.width)
            or np.any(coords[:, 1] < 0)
            or np.any(coords[:, 1] >= image.height)
        ):
            raise ValueError("A point lies outside the image.")

        if not np.all(np.isin(labels, [0, 1])):
            raise ValueError("Point labels must be 1 (keep) or 0 (exclude).")

        predict_args["point_coords"] = coords
        predict_args["point_labels"] = labels

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(**predict_args)

    best_index = int(np.argmax(scores))
    mask = np.asarray(masks[best_index]).astype(bool)
    score = float(scores[best_index])

    return mask, score
'''

if old not in source:
    print("ERROR: Could not find the exact original segment_image function.")
    print("No changes made. Your backup is app_before_point_refinement.py")
    raise SystemExit(1)

path.write_text(source.replace(old, new, 1))
print("Updated segment_image() to support optional point prompts.")
