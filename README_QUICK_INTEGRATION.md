# FastReasonSeg — quick integration package

This package adds a reusable object-level Digital Twin JSON builder, a lightweight
video ID association baseline, and an optional Qwen3 + LoRA inference wrapper.
It is designed to sit beside your existing `app.py`; it does not overwrite it.

## What this package does

1. `fastreasonseg_core/digital_twin.py`: combines per-object labels, confidence,
   bounding boxes, SAM masks, mask geometry, and relative depth statistics.
2. `fastreasonseg_core/pipeline.py`: calls your existing `detect_objects`,
   `segment_image`, and `estimate_depth` functions through callbacks.
3. `fastreasonseg_core/tracking.py`: associates detections across frames and
   preserves track IDs using IoU and center distance. This is a lightweight
   association baseline, **not SAM2 video propagation**.
4. `fastreasonseg_core/qwen_reasoner.py`: optional Qwen3-0.6B adapter inference
   over the generated JSON. This does not train the model or reproduce GRPO.
5. `tests/test_core.py`: synthetic tests for JSON construction and tracking.

## Install into your existing project

Extract this ZIP into `~/Documents/FastReasonSeg` so the `fastreasonseg_core/`
folder and `tests/` folder sit next to `app.py`.

Your existing environment already reported successful imports for PyTorch,
Transformers and SAM2, and MPS availability. Do not reinstall those packages.
If `peft` is missing and you want the optional Qwen adapter, run:

```bash
cd ~/Documents/FastReasonSeg
source .venv/bin/activate
pip install 'peft>=0.13'
```

## Integrate the Digital Twin button into app.py

### 1. Add this import near the other imports

```python
from fastreasonseg_core.pipeline import run_image_digital_twin
```

### 2. Add this block inside the Image Segmentation page, after the existing
mask display/results block (inside the `if page == "Image Segmentation":` area).

```python
st.divider()
st.subheader("Digital Twin Construction")

if st.button("Build Digital Twin JSON", key="build_digital_twin"):
    if saved_image is None or not prompt.strip():
        st.warning("Upload an image and enter an object prompt first.")
    else:
        try:
            with st.spinner("Building object-level digital twin..."):
                twin, twin_masks = run_image_digital_twin(
                    image=saved_image,
                    prompt=prompt.strip(),
                    threshold=threshold,
                    detect_fn=detect_objects,
                    segment_fn=segment_image,
                    depth_fn=estimate_depth,
                )
            st.session_state["digital_twin_json"] = twin
            st.session_state["digital_twin_masks"] = twin_masks
            st.success(f"Digital twin created for {twin['object_count']} object(s).")
        except Exception as exc:
            st.error(f"Digital twin construction failed: {exc}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())

if "digital_twin_json" in st.session_state:
    twin = st.session_state["digital_twin_json"]
    st.json(twin)
    st.download_button(
        "Download Digital Twin JSON",
        data=json.dumps(twin, indent=2),
        file_name="digital_twin.json",
        mime="application/json",
        key="download_digital_twin_json",
    )
```

**Note:** This integration re-runs OWLv2, SAM2 and Depth Anything when the button
is pressed. It currently uses the prompt as the object label because the existing
`detect_objects()` returns only boxes and confidence scores; it discards OWLv2's
class-label IDs. The module accepts explicit labels if you later extend the
existing detector to return them.

## Run tests

```bash
cd ~/Documents/FastReasonSeg
source .venv/bin/activate
python -m unittest discover -s tests -v
```

## Optional Qwen reasoning

After building a `twin` dictionary in the app:

```python
from fastreasonseg_core.qwen_reasoner import QwenDigitalTwinReasoner

@st.cache_resource(show_spinner=False)
def load_qwen_reasoner():
    return QwenDigitalTwinReasoner(
        base_model="Qwen/Qwen3-0.6B",
        adapter_path="models/final_adapter",
        device=DEVICE,
    )

question = st.text_input("Ask about this scene", "Which object is largest?")
if st.button("Ask Qwen", key="ask_qwen"):
    try:
        with st.spinner("Reasoning over digital twin..."):
            answer = load_qwen_reasoner().answer(twin, question)
        st.write(answer)
    except Exception as exc:
        st.error(f"Qwen inference failed: {exc}")
```

This is adapter inference only. It does not verify adapter quality or reproduce the
paper's teacher GRPO, student SFT/GRPO, or distillation procedure.

## Limitations you must disclose in a report/demo

- Depth Anything V2 output is relative depth, not metric distance.
- Current app's video path re-detects and re-segments each frame independently.
- The included IoU tracker is a baseline for stable IDs; it is not SAM2's video
  memory/propagation mechanism and may fail under occlusion or large motion.
- Existing LoRA configuration is rank 8 / alpha 16; the paper configuration
  previously extracted for its student is rank 64 / alpha 128.
- This package adds an integration layer and tests. It does **not** make the
  project a complete reproduction of the CVPR paper or official RVTBench results.
