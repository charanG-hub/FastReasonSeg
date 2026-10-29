import io
import json
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    Owlv2ForObjectDetection,
    Owlv2Processor,
)
from sam2.sam2_image_predictor import SAM2ImagePredictor
from streamlit_drawable_canvas import st_canvas

# =========================================================
# FASTREASONSEG - IMAGE AND VIDEO SEGMENTATION WEB APP
# =========================================================

st.set_page_config(
    page_title="FastReasonSeg",
    page_icon="🔍",
    layout="wide",
)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

OWL_MODEL_ID = "google/owlv2-base-patch16-ensemble"
SAM_MODEL_ID = "facebook/sam2.1-hiera-small"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_owl():
    processor = Owlv2Processor.from_pretrained(OWL_MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL_ID)
    model.to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def load_sam():
    predictor = SAM2ImagePredictor.from_pretrained(
        SAM_MODEL_ID,
        device=DEVICE,
    )
    return predictor

@st.cache_resource(show_spinner=False)
def load_depth():
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
    model.to(DEVICE)
    model.eval()
    return processor, model


# =========================================================
# OBJECT DETECTION WITH TEXT PROMPT
# =========================================================

def detect_objects(image, prompt, threshold=0.15):
    """Detect objects with multiple text descriptions using OWLv2."""

    processor, model = load_owl()
    image = image.convert("RGB")

    prompt = prompt.strip().lower()

    # Try multiple descriptions to improve text-guided detection.
    if prompt in {"person", "people", "human"}:
        candidate_labels = [
            "person",
            "a person walking",
            "a woman walking",
            "a man walking",
            "a human body",
        ]
    else:
        candidate_labels = [
            prompt,
            f"a {prompt}",
        ]

    inputs = processor(
        text=[candidate_labels],
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor([
        [image.height, image.width]
    ])

    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=float(threshold),
    )

    result = results[0]

    boxes = result["boxes"].detach().cpu().numpy()
    scores = result["scores"].detach().cpu().numpy()
    labels = result["labels"].detach().cpu().numpy()

    if len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    # Keep coordinates within image boundaries.
    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]], 0, image.width
    )
    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]], 0, image.height
    )

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights

    valid = (
        (widths > 0)
        & (heights > 0)
        & (areas >= image.width * image.height * 0.002)
    )

    boxes = boxes[valid]
    scores = scores[valid]
    labels = labels[valid]

    if len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    # Sort by confidence, highest first.
    order = np.argsort(scores)[::-1]

    boxes = boxes[order]
    scores = scores[order]

    # Remove highly overlapping duplicate detections.
    keep = []

    for i, box in enumerate(boxes):
        duplicate = False

        for j in keep:
            x1 = max(box[0], boxes[j][0])
            y1 = max(box[1], boxes[j][1])
            x2 = min(box[2], boxes[j][2])
            y2 = min(box[3], boxes[j][3])

            intersection = max(0, x2 - x1) * max(0, y2 - y1)

            area1 = (
                (box[2] - box[0]) *
                (box[3] - box[1])
            )
            area2 = (
                (boxes[j][2] - boxes[j][0]) *
                (boxes[j][3] - boxes[j][1])
            )

            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0

            if iou > 0.70:
                duplicate = True
                break

        if not duplicate:
            keep.append(i)

    boxes = boxes[keep]
    scores = scores[keep]

    return boxes.astype(np.float32), scores.astype(np.float32)

# =========================================================
# IMAGE SEGMENTATION WITH SAM 2.1
# =========================================================

def segment_image(image, box):
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


# =========================================================
# OVERLAY GENERATION
# =========================================================

def create_overlay(image, mask, box=None):
    rgb = np.asarray(image.convert("RGB")).copy()

    green = np.array([0, 220, 100], dtype=np.float32)

    rgb[mask] = (
        0.55 * rgb[mask].astype(np.float32)
        + 0.45 * green
    ).astype(np.uint8)

    output = Image.fromarray(rgb)

    if box is not None:
        draw = ImageDraw.Draw(output)

        draw.rectangle(
            [float(v) for v in box],
            outline=(255, 50, 40),
            width=4,
        )

    return output


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# =========================================================
# DEPTH ESTIMATION
# =========================================================

def estimate_depth(image):
    processor, model = load_depth()

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    depth = torch.nn.functional.interpolate(
        outputs.predicted_depth.unsqueeze(1),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze().detach().float().cpu().numpy()

    low, high = np.percentile(depth, [2, 98])

    normalized = np.clip(
        (depth - low) / max(float(high - low), 1e-8),
        0,
        1,
    )

    depth_image = Image.fromarray(
        (normalized * 255).astype(np.uint8)
    )

    statistics = {
        "mean_output": float(depth.mean()),
        "median_output": float(np.median(depth)),
        "minimum_output": float(depth.min()),
        "maximum_output": float(depth.max()),
        "standard_deviation": float(depth.std()),
    }

    return depth_image, statistics


# =========================================================
# VIDEO SEGMENTATION
# =========================================================

def process_video(
    video_bytes,
    suffix,
    prompt,
    threshold,
    max_frames,
    progress,
):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        input_path = temp_dir / (
            "input" + (suffix or ".mp4")
        )

        output_path = temp_dir / "segmented.mp4"

        input_path.write_bytes(video_bytes)

        capture = cv2.VideoCapture(str(input_path))

        if not capture.isOpened():
            raise RuntimeError(
                "Cannot open this video. Try an MP4 video."
            )

        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        frames_limit = min(
            total_frames if total_frames > 0 else max_frames,
            max_frames,
        )

        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("Invalid video dimensions.")

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            capture.release()
            raise RuntimeError(
                "Could not create the output video."
            )

        processed = 0
        detected_frames = 0

        try:
            while processed < frames_limit:
                success, frame = capture.read()

                if not success:
                    break

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                frame_image = Image.fromarray(rgb)

                boxes, scores = detect_objects(
                    frame_image,
                    prompt,
                    threshold,
                )

                if len(boxes) > 0:
                    index = int(np.argmax(scores))
                    box = boxes[index]

                    mask, _ = segment_image(
                        frame_image,
                        box,
                    )

                    detected_frames += 1

                    if mask.shape != (height, width):
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)

                    rgb[mask] = (
                        0.55 * rgb[mask].astype(np.float32)
                        + 0.45
                        * np.array(
                            [0, 220, 100],
                            dtype=np.float32,
                        )
                    ).astype(np.uint8)

                    x1, y1, x2, y2 = [
                        int(v) for v in box
                    ]

                    cv2.rectangle(
                        rgb,
                        (x1, y1),
                        (x2, y2),
                        (255, 50, 40),
                        2,
                    )

                writer.write(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                )

                processed += 1

                progress.progress(
                    min(processed / max(1, frames_limit), 1.0)
                )

        finally:
            capture.release()
            writer.release()

        if processed == 0:
            raise RuntimeError("No frames were processed.")

        output_bytes = output_path.read_bytes()

        metadata = {
            "prompt": prompt,
            "frames_processed": processed,
            "frames_with_detection": detected_frames,
            "fps": float(fps),
            "width": width,
            "height": height,
            "device": DEVICE,
        }

        return output_bytes, metadata


# =========================================================
# WEBSITE SIDEBAR
# =========================================================

st.title("🔍 FastReasonSeg")

st.write(
    "Prompt-guided image segmentation, depth estimation "
    "and experimental video segmentation."
)

with st.sidebar:
    st.header("Navigation")

    page = st.radio(
        "Select feature",
        [
            "Image Segmentation",
            "Depth Analysis",
            "Video Segmentation",
            "About",
        ],
    )

    st.divider()

    st.write("Compute device:", f"`{DEVICE.upper()}`")

    threshold = st.slider(
        "Detection threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.15,
        step=0.05,
        help="Lower this if an object is not detected.",
    )


# =========================================================
# IMAGE SEGMENTATION PAGE
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

                    # Measure pairwise overlap before combining masks.
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

# DEPTH ANALYSIS PAGE
# =========================================================

elif page == "Depth Analysis":

    st.header("Depth Analysis")

    st.info(
        "Depth Anything V2 produces relative depth estimates, "
        "not calibrated distances in meters."
    )

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="depth_upload",
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption="Original image",
            use_container_width=True,
        )

        if st.button(
            "Estimate depth",
            type="primary",
        ):
            try:
                with st.spinner(
                    "Estimating relative depth..."
                ):
                    depth_image, statistics = estimate_depth(
                        image
                    )

                col1, col2 = st.columns(2)

                col1.image(
                    image,
                    caption="Original",
                    use_container_width=True,
                )

                col2.image(
                    depth_image,
                    caption="Relative depth map",
                    use_container_width=True,
                )

                st.json(statistics)

                st.download_button(
                    "Download depth map",
                    image_to_png_bytes(depth_image),
                    "depth_map.png",
                    "image/png",
                )

            except Exception as exc:
                st.error(f"Depth estimation failed: {exc}")
                st.code(traceback.format_exc())


# =========================================================
# VIDEO SEGMENTATION PAGE
# =========================================================

elif page == "Video Segmentation":

    st.header("Video Segmentation")

    st.warning(
        "This is frame-by-frame segmentation, not persistent "
        "SAM 2 video-object tracking. Processing may be slow."
    )

    uploaded = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "avi"],
        key="video_upload",
    )

    prompt = st.text_input(
        "Object to segment",
        placeholder="dog, car, person, truck",
        key="video_prompt",
    )

    max_frames = st.slider(
        "Maximum frames",
        min_value=10,
        max_value=120,
        value=30,
        step=10,
    )

    if uploaded is not None:
        st.video(uploaded)

        if st.button(
            "Process video",
            type="primary",
        ):
            if not prompt.strip():
                st.warning(
                    "Please enter an object prompt."
                )

            else:
                progress = st.progress(0.0)

                try:
                    suffix = (
                        Path(uploaded.name).suffix.lower()
                        or ".mp4"
                    )

                    with st.spinner(
                        "Processing video frames..."
                    ):
                        output_bytes, metadata = process_video(
                            uploaded.getvalue(),
                            suffix,
                            prompt.strip(),
                            threshold,
                            max_frames,
                            progress,
                        )

                    st.success(
                        "Video processing completed."
                    )

                    st.video(output_bytes)

                    st.json(metadata)

                    st.download_button(
                        "Download segmented video",
                        output_bytes,
                        "segmented_video.mp4",
                        "video/mp4",
                    )

                    st.download_button(
                        "Download video metadata",
                        json.dumps(metadata, indent=2),
                        "video_metadata.json",
                        "application/json",
                    )

                except Exception as exc:
                    st.error(
                        f"Video processing failed: {exc}"
                    )
                    st.code(traceback.format_exc())


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About":

    st.header("About FastReasonSeg")

    st.markdown(
        """
        ### Models

        - **OWLv2:** Detects objects from text prompts.
        - **SAM 2.1 Hiera Small:** Produces segmentation masks.
        - **Depth Anything V2 Small:** Estimates relative depth.

        ### Features

        - Prompt-based object detection.
        - Image segmentation and mask visualization.
        - Downloadable masks and metadata.
        - Relative depth maps.
        - Experimental frame-by-frame video segmentation.

        ### Research limitations

        This is a prototype inspired by the FastReasonSeg project.
        It is not a full reproduction of the research paper.

        It does not implement the complete reasoning teacher,
        GRPO training, or student distillation pipeline.

        Detection and segmentation quality depend on the image,
        prompt, thresholds, and pretrained model behavior.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "FastReasonSeg | Research prototype | "
    "OWLv2 + SAM 2.1 + Depth Anything V2"
)