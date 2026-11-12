import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3-0.6B"
ADAPTER = Path("models/final_adapter")
OUT = Path("results/base_vs_adapter.json")

TESTS = [
    {
        "id": "object_grounding",
        "category": "Object grounding",
        "prompt": (
            "Scene description: A red car is parked on the left. "
            "A blue truck is parked on the right. "
            "Question: Which object is on the right? "
            "Answer using the provided scene only."
        ),
        "expected": "blue truck",
    },
    {
        "id": "segmentation_plan",
        "category": "Segmentation planning",
        "prompt": (
            "A detector returns a bounding box for a dog in an image. "
            "Describe a concise, ordered pipeline using a segmentation "
            "model to obtain a binary mask. Return four numbered steps. "
            "Do not claim that you have processed an actual image."
        ),
        "expected": "ordered pipeline with a mask output",
    },
    {
        "id": "temporal_reasoning",
        "category": "Temporal reasoning",
        "prompt": (
            "A tracked object's center is at x=100 in frame 1, "
            "x=115 in frame 2, and x=130 in frame 3. "
            "Assuming the same image coordinate system, describe its "
            "horizontal movement and the displacement per frame."
        ),
        "expected": "rightward; 15 pixels per frame",
    },
    {
        "id": "spatial_reasoning",
        "category": "Spatial reasoning",
        "prompt": (
            "Object A has bounding box [10, 20, 50, 60]. "
            "Object B has bounding box [70, 20, 110, 60]. "
            "Coordinates are [x1, y1, x2, y2]. "
            "Which object is farther to the right? Explain briefly."
        ),
        "expected": "Object B",
    },
    {
        "id": "uncertainty",
        "category": "Uncertainty handling",
        "prompt": (
            "You are given no image, no video, and no detector output. "
            "The user asks: What objects are in the scene? "
            "Answer accurately without inventing observations."
        ),
        "expected": "states that the scene cannot be determined",
    },
]


def generate(model, tokenizer, device, prompt):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant for computer vision research. "
                "Use only the supplied information. Be concise and "
                "do not invent observations."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    if not ADAPTER.exists():
        raise FileNotFoundError(f"Adapter directory not found: {ADAPTER}")

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("BASE MODEL vs EXISTING LORA ADAPTER")
    print("=" * 60)
    print("Device:", device)
    print("Tests:", len(TESTS))
    print("Adapter:", ADAPTER)
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(BASE)

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    base.eval()

    results = []

    for test in TESTS:
        print(f"\n[{test['id']}] Base model generating...")
        start = time.time()
        base_answer = generate(base, tokenizer, device, test["prompt"])
        base_seconds = round(time.time() - start, 2)

        results.append({
            **test,
            "base_output": base_answer,
            "base_seconds": base_seconds,
        })

        print("Base:", base_answer[:350])

    print("\nLoading LoRA adapter...")
    adapted = PeftModel.from_pretrained(
        base,
        str(ADAPTER),
        is_trainable=False,
    )
    adapted.eval()

    for result in results:
        print(f"\n[{result['id']}] Adapter generating...")
        start = time.time()
        result["adapter_output"] = generate(
            adapted, tokenizer, device, result["prompt"]
        )
        result["adapter_seconds"] = round(time.time() - start, 2)

        print("Adapter:", result["adapter_output"][:350])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "base_model": BASE,
                "adapter_path": str(ADAPTER),
                "device": device,
                "test_count": len(results),
                "results": results,
                "evaluation_note": (
                    "Outputs require human review. This text-only test "
                    "does not measure image/video segmentation accuracy."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("Saved:", OUT)
    print("Tests completed:", len(results))
    print("Compare base_output and adapter_output for each test.")
    print("This is a small diagnostic set, not a benchmark.")
    print("=" * 60)


if __name__ == "__main__":
    main()
