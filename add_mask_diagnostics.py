from pathlib import Path

path = Path("app.py")
source = path.read_text()

old = '''                    for mask in valid_masks.values():
                        combined_mask |= mask
'''

new = '''                    # Measure pairwise overlap before combining masks.
                    st.subheader("Mask overlap diagnostics")

                    indices = list(valid_masks.keys())
                    overlap_found = False

                    for pos, index_a in enumerate(indices):
                        mask_a = valid_masks[index_a]
                        area_a = int(mask_a.sum())

                        for index_b in indices[pos + 1:]:
                            mask_b = valid_masks[index_b]
                            area_b = int(mask_b.sum())
                            intersection = int(np.logical_and(mask_a, mask_b).sum())

                            smaller_area = min(area_a, area_b)
                            overlap_of_smaller = (
                                100.0 * intersection / smaller_area
                                if smaller_area > 0 else 0.0
                            )

                            union = int(np.logical_or(mask_a, mask_b).sum())
                            iou = (
                                100.0 * intersection / union
                                if union > 0 else 0.0
                            )

                            if overlap_of_smaller >= 20.0:
                                overlap_found = True
                                st.warning(
                                    f"Candidate {index_a + 1} and Candidate "
                                    f"{index_b + 1} overlap substantially: "
                                    f"{overlap_of_smaller:.1f}% of the smaller "
                                    f"mask overlaps. Mask IoU: {iou:.1f}%."
                                )
                            else:
                                st.caption(
                                    f"Candidate {index_a + 1} vs "
                                    f"Candidate {index_b + 1}: "
                                    f"{overlap_of_smaller:.1f}% of the smaller "
                                    f"mask overlaps; mask IoU {iou:.1f}%."
                                )

                    if not overlap_found:
                        st.success(
                            "No substantial pairwise mask overlap was detected "
                            "among the selected candidates."
                        )

                    for mask in valid_masks.values():
                        combined_mask |= mask
'''

count = source.count(old)

if count != 1:
    raise SystemExit(
        f"Expected exactly one mask-combination block, found {count}. "
        "No changes were made."
    )

path.write_text(source.replace(old, new, 1))
print("Added pairwise mask-overlap diagnostics to app.py.")
