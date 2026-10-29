from pathlib import Path
import re
import shutil

path = Path("app.py")
source = path.read_text()

# Rebuild each candidate's display from the latest session-state mask.
old = '''                    for index, mask in valid_masks.items():
                        mask_score = mask_scores_by_index.get(index, 0.0)
                        individual_overlay = create_overlay(
                            saved_image,
                            mask,
                            box=boxes[index],
                        )
                        individual_mask_image = Image.fromarray(
                            mask.astype(np.uint8) * 255,
                            mode="L",
                        )
                        area_percent = float(mask.mean() * 100)
'''

new = '''                    for index, _mask in valid_masks.items():
                        # Always use the newest mask stored in session state.
                        current_masks = st.session_state.get(
                            "multi_candidate_masks", {}
                        )
                        mask = current_masks.get(index, _mask)
                        mask = np.asarray(mask).astype(bool)

                        current_scores = st.session_state.get(
                            "multi_candidate_mask_scores", {}
                        )
                        mask_score = float(
                            current_scores.get(
                                index, mask_scores_by_index.get(index, 0.0)
                            )
                        )

                        individual_overlay = create_overlay(
                            saved_image,
                            mask,
                            box=boxes[index],
                        )
                        individual_mask_image = Image.fromarray(
                            mask.astype(np.uint8) * 255,
                            mode="L",
                        )
                        area_percent = float(mask.mean() * 100)
'''

if old not in source:
    raise SystemExit(
        "PATCH STOPPED: expected code block not found. app.py was not changed."
    )

backup = Path("app_before_refined_mask_display_fix.py")
shutil.copy2(path, backup)
path.write_text(source.replace(old, new, 1))

print("Patch applied. Backup:", backup)
