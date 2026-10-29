from pathlib import Path
import shutil

path = Path("app.py")
source = path.read_text()

start_marker = "    # When point prompts are supplied, prefer masks that respect them."
end_marker = "    return mask, score"

start = source.find(start_marker)
end = source.find(end_marker, start)

if start == -1 or end == -1:
    raise SystemExit(
        "Could not locate the expected SAM2 mask-selection code. "
        "No changes made."
    )

end += len(end_marker)

replacement = '''    # Select the SAM2 candidate using prompt compliance first,
    # then model confidence and mask stability.
    if point_coords is not None and len(point_coords) > 0:
        points = np.asarray(point_coords, dtype=np.int32)
        labels = np.asarray(point_labels, dtype=np.int32)

        quality = np.asarray(scores, dtype=np.float32).copy()
        violations = np.zeros(len(masks), dtype=np.int32)

        for candidate_index, candidate_mask in enumerate(masks):
            candidate_mask = np.asarray(candidate_mask).astype(bool)

            for (x, y), label in zip(points, labels):
                inside = bool(candidate_mask[y, x])

                if label == 1 and not inside:
                    violations[candidate_index] += 1
                elif label == 0 and inside:
                    violations[candidate_index] += 1

        # A candidate respecting all points always outranks one that
        # violates a point. Confidence breaks ties among valid candidates.
        best_index = int(
            np.lexsort((quality, -violations))[0]
        )

        selected_mask = np.asarray(masks[best_index]).astype(bool)

        # Enforce the point labels on the returned mask. This guarantees
        # the clicked positive pixels are included and negative pixels excluded.
        for (x, y), label in zip(points, labels):
            selected_mask[y, x] = bool(label)

        mask = selected_mask
    else:
        best_index = int(np.argmax(scores))
        mask = np.asarray(masks[best_index]).astype(bool)

    score = float(scores[best_index])
    return mask, score'''

backup = Path("app_before_improved_mask_selection.py")
shutil.copy2(path, backup)
path.write_text(source[:start] + replacement + source[end:])

print("Updated SAM2 point-guided mask selection.")
print(f"Backup saved as: {backup}")
