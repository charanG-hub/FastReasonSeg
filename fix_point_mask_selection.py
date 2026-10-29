from pathlib import Path
import shutil

path = Path("app.py")
source = path.read_text()

old = '''    best_index = int(np.argmax(scores))
    mask = np.asarray(masks[best_index]).astype(bool)
    score = float(scores[best_index])

    return mask, score
'''

new = '''    # When point prompts are supplied, prefer masks that respect them.
    if point_coords is not None and len(point_coords) > 0:
        points = np.asarray(point_coords, dtype=np.int32)
        labels = np.asarray(point_labels, dtype=np.int32)

        candidate_quality = np.asarray(scores, dtype=np.float32).copy()

        for candidate_index, candidate_mask in enumerate(masks):
            candidate_mask = np.asarray(candidate_mask).astype(bool)

            for (x, y), label in zip(points, labels):
                inside = bool(candidate_mask[y, x])

                # Strongly penalize a mask that violates a point prompt.
                if label == 1 and not inside:
                    candidate_quality[candidate_index] -= 2.0
                elif label == 0 and inside:
                    candidate_quality[candidate_index] -= 2.0

        best_index = int(np.argmax(candidate_quality))
    else:
        best_index = int(np.argmax(scores))

    mask = np.asarray(masks[best_index]).astype(bool)
    score = float(scores[best_index])

    return mask, score
'''

if old not in source:
    raise SystemExit(
        "Patch stopped: the expected mask-selection code was not found. "
        "app.py was not changed."
    )

shutil.copy2(path, "app_before_point_selection_fix.py")
path.write_text(source.replace(old, new, 1))
print("Updated mask selection. Backup: app_before_point_selection_fix.py")
