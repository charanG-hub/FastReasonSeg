from pathlib import Path

app_path = Path("app.py")
source = app_path.read_text()

start_marker = "# IMAGE SEGMENTATION PAGE"
end_marker = "# DEPTH ANALYSIS PAGE"

start = source.index(start_marker)
end = source.index(end_marker, start)

replacement = '''# IMAGE SEGMENTATION PAGE
# =========================================================

if page == "Image Segmentation":

    st.header("Image Segmentation")
    st.caption("Detect multiple objects and segment your selected candidates.")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    prompt = st.text_input(
        "Enter object prompt",
        placeholder="person, dog, truck, car",
        key="image_prompt",
    )

    threshold = st.slider(
        "Detection threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.15,
        step=0.05,
        key="image_detection_threshold",
        help="Lower this if an object is not detected.",
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Original image", use_container_width=True)

        if st.button("Detect candidates", type="primary", key="detect_candidates"):
            if not prompt.strip():
                st.warning("Enter an object prompt first.")
            else:
                try:
                    with st.spinner("Detecting objects with OWLv2..."):
                        boxes, scores = detect_objects(image, prompt.strip(), threshold)

                    st.session_state["multi_candidate_image"] = image
                    st.session_state["multi_candidate_prompt"] = prompt.strip()
                    st.session_state["multi_candidate_boxes"] = boxes
                    st.session_state["multi_candidate_scores"] = scores
                    st.session_state["multi_candidate_upload_name"] = uploaded.name
                    st.session_state["multi_candidate_signature"] = (
                        uploaded.name, uploaded.size, prompt.strip(), threshold
                    )
                    st.session_state.pop("multi_candidate_masks", None)

                except Exception as exc:
                    st.error(f"Detection failed: {exc}")
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())

    # Display the latest detected candidates.
    boxes = st.session_state.get("multi_candidate_boxes")
    scores = st.session_state.get("multi_candidate_scores")
    saved_image = st.session_state.get("multi_candidate_image")

    if boxes is not None and saved_image is not None:
        if len(boxes) == 0:
            st.warning(
                "No valid objects were detected. Try a more specific prompt "
                "or lower the detection threshold."
            )
        else:
            st.subheader(f"Detected candidates: {len(boxes)}")

            detection_options = [
                f"Candidate {i + 1} — confidence {float(scores[i]):.3f}"
                for i in range(len(boxes))
            ]

            # Default to the highest-confidence candidate.
            selected = st.multiselect(
                "Choose candidates to segment",
                options=list(range(len(boxes))),
                default=[0],
                format_func=lambda i: detection_options[i],
                key="multi_candidate_selection",
                help="Select one candidate, several candidates, or all candidates.",
            )

            button_col1, button_col2 = st.columns(2)
            with button_col1:
                if st.button("Select all", key="select_all_candidates"):
                    st.session_state["multi_candidate_selection"] = list(range(len(boxes)))
                    st.rerun()

            with button_col2:
                if st.button("Clear selection", key="clear_candidate_selection"):
                    st.session_state["multi_candidate_selection"] = []
                    st.rerun()

            # Draw all detector boxes so the user can identify candidates.
            preview = saved_image.copy()
            draw = ImageDraw.Draw(preview)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [float(v) for v in box]
                draw.rectangle((x1, y1, x2, y2), outline=(255, 60, 60), width=3)
                draw.text(
                    (x1 + 4, max(0, y1 + 4)),
                    f"{i + 1}: {float(scores[i]):.2f}",
                    fill=(255, 255, 0),
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )

            st.image(
                preview,
                caption="Detection boxes: candidate number and confidence",
                use_container_width=True,
            )

            if st.button(
                "Segment selected candidates",
                type="primary",
                key="segment_selected_candidates",
                disabled=len(selected) == 0,
            ):
                masks_by_index = {}
                mask_scores_by_index = {}

                try:
                    with st.spinner(
                        f"Segmenting {len(selected)} candidate(s) with SAM 2.1..."
                    ):
                        for index in selected:
                            mask, mask_score = segment_image(
                                saved_image,
                                boxes[index],
                            )

                            mask = np.asarray(mask).astype(bool)

                            if mask.shape != (saved_image.height, saved_image.width):
                                raise ValueError(
                                    f"Candidate {index + 1} mask has unexpected shape "
                                    f"{mask.shape}."
                                )

                            masks_by_index[index] = mask
                            mask_scores_by_index[index] = float(mask_score)

                    st.session_state["multi_candidate_masks"] = masks_by_index
                    st.session_state["multi_candidate_mask_scores"] = mask_scores_by_index
                    st.success(f"Segmented {len(masks_by_index)} candidate(s).")

                except Exception as exc:
                    st.error(f"Segmentation failed: {exc}")
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())

            masks_by_index = st.session_state.get("multi_candidate_masks", {})
            mask_scores_by_index = st.session_state.get(
                "multi_candidate_mask_scores", {}
            )

            if masks_by_index:
                # Keep only masks corresponding to the current detected candidates.
                valid_masks = {
                    i: m for i, m in masks_by_index.items()
                    if i < len(boxes)
                }

                if valid_masks:
                    combined_mask = np.zeros(
                        (saved_image.height, saved_image.width),
                        dtype=bool,
                    )

                    for mask in valid_masks.values():
                        combined_mask |= mask

                    combined_overlay = create_overlay(
                        saved_image,
                        combined_mask,
                        box=None,
                    )

                    combined_mask_image = Image.fromarray(
                        (combined_mask.astype(np.uint8) * 255),
                        mode="L",
                    )

                    st.subheader("Combined segmentation")
                    st.caption(
                        "The combined mask is the union of the selected masks. "
                        "Overlapping pixels are included only once."
                    )

                    combined_col1, combined_col2 = st.columns(2)
                    with combined_col1:
                        st.image(
                            combined_overlay,
                            caption="Combined mask overlay",
                            use_container_width=True,
                        )
                    with combined_col2:
                        st.image(
                            combined_mask_image,
                            caption="Combined binary mask",
                            use_container_width=True,
                        )

                    combined_area = float(combined_mask.mean() * 100)
                    st.metric("Combined segmented area", f"{combined_area:.2f}%")

                    st.download_button(
                        "Download combined mask (PNG)",
                        data=(
                            lambda buffer: (
                                combined_mask_image.save(buffer, format="PNG"),
                                buffer.getvalue(),
                            )[1]
                        )(io.BytesIO()),
                        file_name="combined_segmentation_mask.png",
                        mime="image/png",
                        key="download_combined_mask",
                    )

                    overlay_buffer = io.BytesIO()
                    combined_overlay.save(overlay_buffer, format="PNG")
                    st.download_button(
                        "Download combined overlay (PNG)",
                        data=overlay_buffer.getvalue(),
                        file_name="combined_segmentation_overlay.png",
                        mime="image/png",
                        key="download_combined_overlay",
                    )

                    st.subheader("Individual candidate results")

                    for index, mask in valid_masks.items():
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

                        with st.expander(
                            f"Candidate {index + 1} — "
                            f"detection {float(scores[index]):.3f}, "
                            f"mask score {mask_score:.3f}"
                        ):
                            result_col1, result_col2 = st.columns(2)

                            with result_col1:
                                st.image(
                                    individual_overlay,
                                    caption=f"Candidate {index + 1} overlay",
                                    use_container_width=True,
                                )

                            with result_col2:
                                st.image(
                                    individual_mask_image,
                                    caption="Individual binary mask",
                                    use_container_width=True,
                                )

                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            metric_col1.metric(
                                "Detection confidence",
                                f"{float(scores[index]):.3f}",
                            )
                            metric_col2.metric("SAM mask score", f"{mask_score:.3f}")
                            metric_col3.metric("Image area", f"{area_percent:.2f}%")

                            mask_buffer = io.BytesIO()
                            individual_mask_image.save(mask_buffer, format="PNG")
                            st.download_button(
                                f"Download candidate {index + 1} mask",
                                data=mask_buffer.getvalue(),
                                file_name=f"candidate_{index + 1}_mask.png",
                                mime="image/png",
                                key=f"download_candidate_mask_{index}",
                            )

                            individual_buffer = io.BytesIO()
                            individual_overlay.save(individual_buffer, format="PNG")
                            st.download_button(
                                f"Download candidate {index + 1} overlay",
                                data=individual_buffer.getvalue(),
                                file_name=f"candidate_{index + 1}_overlay.png",
                                mime="image/png",
                                key=f"download_candidate_overlay_{index}",
                            )

'''

new_source = source[:start] + replacement + source[end:]

# Validate that the replacement preserves the next page boundary.
if new_source.count(end_marker) != source.count(end_marker):
    raise RuntimeError("Depth Analysis section boundary changed unexpectedly.")

app_path.write_text(new_source)
print("Updated Image Segmentation page successfully.")
print("Original backup: app_backup.py")
