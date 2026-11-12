"""Enhanced Qwen3 LoRA inference and reasoning module with 3D Spatial-Depth Gating.

Provides:
- 3D spatial relational grounding in conversational reasoning prompts.
- Fast candidate re-ranking via SpatialDepthGate.
- Uncertainty-guided boundary refinement integration.
"""
from __future__ import annotations
import json
from typing import Any, Mapping, Optional
from .spatial_depth_gate import SpatialDepthGate
from .boundary_refiner import AdaptiveBoundaryRefiner


class QwenDigitalTwinReasoner:
    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-0.6B",
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 220,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.max_new_tokens = int(max_new_tokens)
        self.spatial_gate = SpatialDepthGate()
        self.boundary_refiner = AdaptiveBoundaryRefiner()

    def answer(self, twin: Mapping[str, Any], question: str) -> str:
        spatial_graph = twin.get("spatial_3d_graph", {})
        relational_summary = spatial_graph.get("relational_prompt_summary", "")
        depth_order = spatial_graph.get("depth_ordering", [])

        compact = {
            "image": twin.get("image"),
            "objects": [{
                "object_id": o.get("object_id"),
                "label": o.get("label"),
                "confidence": o.get("detection_confidence"),
                "box": o.get("bounding_box_xyxy"),
                "mask_area_fraction": o.get("mask", {}).get("area_fraction_of_image"),
                "relative_depth": o.get("relative_depth"),
            } for o in twin.get("objects", [])],
            "depth_ordering_foreground_to_background": depth_order,
            "spatial_3d_relations": relational_summary,
        }
        messages = [
            {"role": "system", "content": (
                "You are an expert visual-scene reasoning assistant. Use the supplied structured scene data, "
                "including 3D spatial relations, relative depth rankings, and bounding boxes. "
                "Disambiguate referring objects accurately. Keep the answer concise and precise."
            )},
            {"role": "user", "content": "Scene JSON:\n" + json.dumps(compact) + "\nQuestion: " + question},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            generated[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

    def select_object(self, twin: Mapping[str, Any], referring_query: str) -> dict[str, Any]:
        """Disambiguate and select the best target object using 3D spatial gating."""
        spatial_graph = twin.get("spatial_3d_graph", {})
        objects = twin.get("objects", [])
        if not objects:
            return {"selected_object_id": None, "reason": "No objects"}

        confs = [o.get("detection_confidence", 0.5) for o in objects]
        ranked = self.spatial_gate.score_candidates(referring_query, spatial_graph, confs)
        if ranked:
            best = ranked[0]
            return {
                "selected_object_id": best["object_id"],
                "score": best["composite_score"],
                "rankings": ranked,
            }
        return {"selected_object_id": objects[0].get("object_id"), "score": 0.5, "rankings": []}
