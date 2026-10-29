from pathlib import Path
import shutil

path = Path("app.py")
source = path.read_text()

old = '''                                    updated_masks[index] = refined_mask
                                    updated_scores[index] = float(refined_score)

                                    st.session_state["multi_candidate_masks"] = (
'''

new = '''                                    previous_mask = updated_masks.get(index)
                                    changed_pixels = None
                                    if previous_mask is not None:
                                        previous_mask = np.asarray(previous_mask).astype(bool)
                                        changed_pixels = int(
                                            np.count_nonzero(previous_mask != refined_mask)
                                        )

                                    updated_masks[index] = refined_mask
                                    updated_scores[index] = float(refined_score)

                                    st.session_state["multi_candidate_masks"] = (
'''

if old not in source:
    raise SystemExit("Could not find the expected code. No changes made.")

backup = Path("app_before_mask_diagnostic.py")
if not backup.exists():
    shutil.copy2(path, backup)

source = source.replace(old, new, 1)

old_message = '''                                    st.success(
                                        f"Candidate {index + 1} mask refined. "
                                        "Review the updated result below."
                                    )
'''

new_message = '''                                    if changed_pixels is None:
                                        st.success(
                                            f"Candidate {index + 1} mask saved. "
                                            "No previous mask was available for comparison."
                                        )
                                    else:
                                        total_pixels = refined_mask.size
                                        changed_percent = 100.0 * changed_pixels / total_pixels
                                        st.success(
                                            f"Candidate {index + 1}: "
                                            f"{changed_pixels:,} pixels changed "
                                            f"({changed_percent:.3f}% of the image). "
                                            f"New SAM score: {refined_score:.4f}"
                                        )
'''

if old_message not in source:
    raise SystemExit("Could not find the expected success message. Restore the backup if needed.")

source = source.replace(old_message, new_message, 1)
path.write_text(source)
print("Added mask-change diagnostics.")
