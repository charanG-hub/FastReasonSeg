from pathlib import Path
import shutil

path = Path("app.py")
source = path.read_text()

start_marker = "    # Remove highly overlapping duplicate detections."
end_marker = "    return boxes.astype(np.float32), scores.astype(np.float32)"

start = source.find(start_marker)
end = source.find(end_marker, start)

if start < 0 or end < 0:
    raise SystemExit(
        "Could not find the expected duplicate-suppression block. "
        "app.py was not changed."
    )

end += len(end_marker)

replacement = '''    # Suppress duplicate boxes while retaining nearby distinct people.
    keep = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        duplicate = False

        for j in keep:
            other = boxes[j]
            ox1, oy1, ox2, oy2 = other

            ix1 = max(x1, ox1)
            iy1 = max(y1, oy1)
            ix2 = min(x2, ox2)
            iy2 = min(y2, oy2)

            intersection = (
                max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            )

            other_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
            union = area + other_area - intersection

            iou = intersection / union if union > 0 else 0.0
            smaller_area = min(area, other_area)
            overlap_smaller = (
                intersection / smaller_area if smaller_area > 0 else 0.0
            )

            # Strong IoU overlap means these are probably duplicate boxes.
            # Containment alone is used more conservatively because one person
            # can overlap another in crowded scenes.
            if iou > 0.70 or (
                iou > 0.45 and overlap_smaller > 0.85
            ):
                duplicate = True
                break

        if not duplicate:
            keep.append(i)

    boxes = boxes[keep]
    scores = scores[keep]

    return boxes.astype(np.float32), scores.astype(np.float32)'''

backup = Path("app_before_person_detection_fix.py")
shutil.copy2(path, backup)
path.write_text(source[:start] + replacement + source[end:])

print("Detection suppression updated.")
print("Backup saved as:", backup)
