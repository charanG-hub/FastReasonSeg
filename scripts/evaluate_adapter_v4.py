import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/reasoning_eval_v1.jsonl"
ADAPTER_PATH = ROOT / "models/final_adapter"
OUTPUT_PATH = ROOT / "results/base_vs_adapter_v4.json"
BASE_MODEL = "Qwen/Qwen3-0.6B"

SPECS = [
    {"type": "numeric", "expected": 5.0, "tolerance": 0.05},
    {"type": "numeric", "expected": 13.0, "tolerance": 0.05},
    {"type": "numeric", "expected": 10.0, "tolerance": 0.05},
    {"type": "signed_direction", "expected": -30.0, "tolerance": 0.05, "direction": "left"},
    {"type": "choice", "expected": "B"},
    {"type": "choice", "expected": "A"},
    {"type": "choice", "expected": "B"},
    {"type": "choice", "expected": "A"},
    {"type": "numeric", "expected": 0.5, "tolerance": 0.005},
    {"type": "numeric", "expected": 2 * 60 / 180, "tolerance": 0.005},
    {"type": "numeric", "expected": 0.8, "tolerance": 0.005},
    {"type": "two_numeric", "expected": [1.0, 1.0], "tolerance": 0.005},
    {"type": "numeric_direction", "expected": 15.0, "tolerance": 0.05, "direction": "right"},
    {"type": "numeric", "expected": 0.8, "tolerance": 0.005},
    {"type": "numeric", "expected": 90.0, "tolerance": 0.05},
    {"type": "numeric", "expected": 0.0, "tolerance": 0.05},
    {"type": "uncertainty"},
    {"type": "uncertainty"},
    {"type": "uncertainty"},
    {"type": "uncertainty"},
]


def load_tests():
    with DATA_PATH.open(encoding="utf-8") as f:
        tests = [json.loads(line) for line in f if line.strip()]
    if len(tests) != len(SPECS):
        raise ValueError(f"Expected {len(SPECS)} questions, found {len(tests)}.")
    return tests


def make_prompt(tokenizer, question):
    # Use the tokenizer's official chat template when available.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise assistant. Answer the user's question directly. "
                "Do not create multiple-choice options unless asked. "
                "For arithmetic, calculate carefully and state the result. "
                "For object choices, answer Object A or Object B. "
                "For yes/no questions, begin with Yes or No. "
                "Keep your answer concise."
            ),
        },
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Safe fallback if the tokenizer has no usable chat template.
        return (
            "<|im_start|>system\n"
            "Answer the question directly and concisely. Do not invent choices.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{question}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


def generate(model, tokenizer, device, question):
    prompt = make_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def numbers(text):
    return [
        float(x.replace(",", ""))
        for x in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text)
    ]


def score(response, spec):
    found = numbers(response)
    lower = response.lower()
    kind = spec["type"]
    passed = False
    reason = ""

    if kind in ("numeric", "signed_direction", "numeric_direction"):
        expected = spec["expected"]
        numeric_ok = any(
            abs(value - expected) <= spec["tolerance"] for value in found
        )
        direction_ok = True

        if kind == "signed_direction":
            direction_ok = "left" in lower
        elif kind == "numeric_direction":
            direction_ok = "right" in lower

        passed = numeric_ok and direction_ok
        reason = f"Expected {expected}; numbers={found}; direction_ok={direction_ok}"

    elif kind == "two_numeric":
        expected = spec["expected"]
        remaining = list(found)
        matches = []
        for target in expected:
            match_index = next(
                (
                    i for i, value in enumerate(remaining)
                    if abs(value - target) <= spec["tolerance"]
                ),
                None,
            )
            if match_index is None:
                matches.append(False)
            else:
                matches.append(True)
                remaining.pop(match_index)
        passed = all(matches)
        reason = f"Expected two values {expected}; numbers={found}"

    elif kind == "choice":
        expected = spec["expected"]
        # Check for an explicit object choice first.
        choices = re.findall(r"\bobject\s+([ab])\b", lower)
        if choices:
            actual = choices[-1].upper()
        else:
            letters = re.findall(r"\b([ab])\b", lower)
            actual = letters[-1].upper() if letters else None
        passed = actual == expected
        reason = f"Expected Object {expected}; detected={actual}"

    elif kind == "uncertainty":
        # Questions are phrased so the correct direct answer is No.
        passed = bool(re.search(r"^\s*no\b", response, re.IGNORECASE))
        reason = "Expected a direct answer beginning with No."

    return {
        "passed": passed,
        "scoring_type": kind,
        "response": response,
        "numbers_found": found,
        "expected": spec,
        "reason": reason,
    }


def summarize(records, key):
    total = len(records)
    passed = sum(record[key]["passed"] for record in records)
    categories = {}
    for record in records:
        category = record["category"]
        bucket = categories.setdefault(category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(record[key]["passed"])
    for bucket in categories.values():
        bucket["accuracy"] = bucket["passed"] / bucket["total"]
    return {
        "passed": passed,
        "total": total,
        "accuracy": passed / total if total else 0,
        "by_category": categories,
    }


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_PATH}")

    tests = load_tests()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)
    print("Base model:", BASE_MODEL)
    print("Questions:", len(tests))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
    ).to(device)
    base.eval()

    print("Loading LoRA adapter...")
    adapted = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    adapted.eval()

    # Quick prompt-format sanity check before the full evaluation.
    sanity_question = "An object's center moves by dx=3 pixels and dy=4 pixels. What is its displacement magnitude?"
    print("\n===== SANITY CHECK =====")
    print("Base:", generate(base, tokenizer, device, sanity_question)[:500])
    print("Adapter:", generate(adapted, tokenizer, device, sanity_question)[:500])

    records = []
    for index, (test, spec) in enumerate(zip(tests, SPECS), start=1):
        print(f"\n[{index}/{len(tests)}] {test['category']}: {test['question']}")

        # Disable adapter for baseline, then enable it for the comparison.
        adapted.disable_adapter_layers()
        base_response = generate(adapted, tokenizer, device, test["question"])
        adapted.enable_adapter_layers()
        adapter_response = generate(adapted, tokenizer, device, test["question"])

        base_score = score(base_response, spec)
        adapter_score = score(adapter_response, spec)

        records.append({
            "index": index,
            "category": test["category"],
            "question": test["question"],
            "reference_answer": test["answer"],
            "base_output": base_response,
            "adapter_output": adapter_response,
            "base_score": base_score,
            "adapter_score": adapter_score,
        })

        print("Base:", repr(base_response[:220]))
        print("Base score:", base_score["passed"], "-", base_score["reason"])
        print("Adapter:", repr(adapter_response[:220]))
        print("Adapter score:", adapter_score["passed"], "-", adapter_score["reason"])

    payload = {
        "model": BASE_MODEL,
        "adapter_path": str(ADAPTER_PATH),
        "device": device,
        "evaluation_count": len(records),
        "note": (
            "Small authored diagnostic set, not a robust benchmark. "
            "Rule-based scores require manual review. The adapter comparison "
            "does not establish research-paper reproduction."
        ),
        "base_summary": summarize(records, "base_score"),
        "adapter_summary": summarize(records, "adapter_score"),
        "results": records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n===== V4 SUMMARY =====")
    print(json.dumps({
        "base_summary": payload["base_summary"],
        "adapter_summary": payload["adapter_summary"],
        "output": str(OUTPUT_PATH),
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
