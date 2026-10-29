from pathlib import Path

path = Path("app.py")
source = path.read_text()

anchor = '''                            result_col1, result_col2 = st.columns(2)

                            with result_col1:
'''

replacement = '''                            st.markdown("#### Refine this candidate's mask")
                            st.caption(
                                "Enter image pixel coordinates. Add a positive point "
                                "inside the person and an optional negative point on "
                                "background or another person. Coordinates start at "
                                "(0, 0) in the top-left corner."
                            )

                            refine_key = f"refine_points_{index}"
                            if refine_key not in st.session_state:
                                st.session_state[refine_key] = False

                            with st.form(key=f"refine_form_{index}"):
                                point_col1, point_col2 = st.columns(2)

                                with point_col1:
                                    st.markdown("**Positive point (keep)**")
                                    positive_x = st.number_input(
                                        "Positive X",
                                        min_value=0,
                                        max_value=max(0, saved_image.width - 1),
                                        value=int((boxes[index][0] + boxes[index][2]) / 2),
                                        key=f"positive_x_{index}",
                                    )
                                    positive_y = st.number_input(
                                        "Positive Y",
                                        min_value=0,
                                        max_value=max(0, saved_image.height - 1),
                                        value=int((boxes[index][1] + boxes[index][3]) / 2),
                                        key=f"positive_y_{index}",
                                    )

                                with point_col2:
                                    st.markdown("**Negative point (exclude, optional)**")
                                    use_negative = st.checkbox(
                                        "Use negative point",
                                        key=f"use_negative_{index}",
                                    )
                                    negative_x = st.number_input(
                                        "Negative X",
                                        min_value=0,
                                        max_value=max(0, saved_image.width - 1),
                                        value=int(max(0, min(saved_image.width - 1,
                                            boxes[index][2] + 10))),
                                        key=f"negative_x_{index}",
                                        disabled=not use_negative,
                                    )
                                    negative_y = st.number_input(
                                        "Negative Y",
                                        min_value=0,
                                        max_value=max(0, saved_image.height - 1),
                                        value=int((boxes[index][1] + boxes[index][3]) / 2),
                                        key=f"negative_y_{index}",
                                        disabled=not use_negative,
                                    )

                                refine_clicked = st.form_submit_button(
                                    "Refine mask with points",
                                    type="primary",
                                )

                            if refine_clicked:
                                point_coords = [[float(positive_x), float(positive_y)]]
                                point_labels = [1]

                                if use_negative:
                                    point_coords.append([
                                        float(negative_x), float(negative_y)
                                    ])
                                    point_labels.append(0)

                                try:
                                    with st.spinner(
                                        f"Refining candidate {index + 1}..."
                                    ):
                                        refined_mask, refined_score = segment_image(
                                            saved_image,
                                            boxes[index],
                                            point_coords=point_coords,
                                            point_labels=point_labels,
                                        )

                                    refined_mask = np.asarray(refined_mask).astype(bool)
                                    if refined_mask.shape != (
                                        saved_image.height,
                                        saved_image.width,
                                    ):
                                        raise ValueError(
                                            f"Unexpected refined mask shape: "
                                            f"{refined_mask.shape}"
                                        )

                                    updated_masks = dict(
                                        st.session_state.get(
                                            "multi_candidate_masks", {}
                                        )
                                    )
                                    updated_scores = dict(
                                        st.session_state.get(
                                            "multi_candidate_mask_scores", {}
                                        )
                                    )
                                    updated_masks[index] = refined_mask
                                    updated_scores[index] = float(refined_score)

                                    st.session_state["multi_candidate_masks"] = (
                                        updated_masks
                                    )
                                    st.session_state[
                                        "multi_candidate_mask_scores"
                                    ] = updated_scores
                                    st.success(
                                        f"Candidate {index + 1} mask refined. "
                                        "Review the updated result below."
                                    )
                                    st.rerun()

                                except Exception as exc:
                                    st.error(f"Mask refinement failed: {exc}")
                                    with st.expander("Refinement technical details"):
                                        st.code(traceback.format_exc())

                            result_col1, result_col2 = st.columns(2)

                            with result_col1:
'''

if anchor not in source:
    print("ERROR: Could not find the expected individual-results layout.")
    print("No changes made.")
    raise SystemExit(1)

path.write_text(source.replace(anchor, replacement, 1))
print("Added point-based refinement controls to individual candidate results.")
