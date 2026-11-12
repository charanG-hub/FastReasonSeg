import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/reasoning_eval_v1.jsonl"
ADAPTER_PATH = ROOT / "models/final_adapter"
OUTPUT_PATH = ROOT / "results/base_vs_adapter_v3.json"
BASE_MODEL = "Qwen/Qwen3-0.6B"


# Explicit scoring targets for the 20 authored evaluation questions.
# These targets are separate from the answer text, avoiding the V2
# problem of selecting the wrong number from a multi-number answer.
SPECS = [
    {"type": "numeric", "expected": 5.0, "tolerance": 0.05},
    {"type": "numeric", "expected": 13.0, "tolerance": 0.05},
    {"type": "numeric", "expected": 10.0, "tolerance": 0.05},
    {"type": "signed_direction", "expected": -30.0, "tolerance": 0.05,
     "direction": "left"},
    {"type": "choice", "expected": "B"},
    {"type": "choice", "expected": "A"},
    {"type": "choice", "expected": "B"},
    {"type": "choice", "expected": "A"},
    {"type": "numeric", "expected": 0.5, "tolerance": 0.005},
    {"type": "numeric", "expected": 2 * 60 / 180, "tolerance": 0.005},
    {"type": "numeric", "expected": 0.8, "tolerance": 0.005},
    {"type": "two_numeric", "expected": [1.0, 1.0], "tolerance": 0.005},
    {"type": "numeric_direction", "expected": 15.0, "tolerance": 0.05,
     "direction": "right"},
    {"type": "numeric", "expected": 0.8, "tolerance": 0.005},
    {"type": "numeric", "expected": 90.0, "tolerance": 0.05},
    {"type": "numeric", "expected": 0.0, "tolerance": 0.05},
    {"type": "uncertainty"},
    {"type": "uncertainty"},
    {"type": "uncertainty"},
    {"type": "uncertainty"},
]


def load_tests():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Evaluation data not found: {DATA_PATH}")

    with DATA_PATH.open(encoding="utf-8") as f:
        tests = [json.loads(line) for line in f if line.strip()]

    if len(tests) != len(SPECS):
        raise ValueError(
            f"Expected {len(SPECS)} questions but found {len(tests)}. "
            "Review the question order before evaluating."
        )

    return tests


def generate(model, tokenizer, device, question):
    prompt = (
        "Solve the following question carefully. "
        "Give a concise explanation if useful, then finish with exactly "
        "one line beginning FINAL ANSWER:. Put the answer to the question "
        "on that line. For a choice, write Object A or Object B. "
        "For a yes/no question, begin the final answer with Yes or No.\n\n"
        f"Question: {question}\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def answer_text(response):
    """Prefer the explicitly requested final-answer line."""
    matches = re.findall(
        r"FINAL ANSWER\s*:\s*(.*)",
        response,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1].strip()

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else response.strip()


def numbers(text):
    # Supports integers, decimals, and signed values.
    # Does not treat a percent sign as a different numeric value.
    return [
        float(x.replace(",", ""))
        for x in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text)
    ]


def score(response, spec):
    final = answer_text(response)
    lower = final.lower()
    found = numbers(final)
    kind = spec["type"]

    result = {
        "passed": False,
        "scoring_type": kind,
        "final_answer_used": final,
        "numbers_found": found,
        "expected": spec,
        "reason": "",
    }

    if kind in ("numeric", "signed_direction", "numeric_direction"):
        expected = spec["expected"]

        if not found:
            result["reason"] = "No numeric value found in final answer."
            return result

        # Use the first number in the final-answer line, not an arbitrary
        # number from the full explanation.
        value = found[0]
        numeric_ok = abs(value - expected) <= spec["tolerance"]

        if kind == "signed_direction":
            direction_ok = bool(re.search(r"\bleft\b", lower))
            result["passed"] = numeric_ok and direction_ok
            result["reason"] = (
                f"Numeric match={numeric_ok}; left-direction match={direction_ok}."
            )
        elif kind == "numeric_direction":
            direction_ok = bool(re.search(r"\bright\b", lower))
            result["passed"] = numeric_ok and direction_ok
            result["reason"] = (
                f"Numeric match={numeric_ok}; right-direction match={direction_ok}."
            )
        else:
            result["passed"] = numeric_ok
            result["reason"] = (
                f"First final-answer number={value}; expected={expected}."
            )

        return result

    if kind == "two_numeric":
        expected = spec["expected"]
        tol = spec["tolerance"]
        result["passed"] = (
            len(found) >= 2
            and abs(found[0] - expected[0]) <= tol
            and abs(found[1] - expected[1]) <= tol
        )
        result["reason"] = (
            f"Expected first two final-answer numbers to be {expected}; "
            f"found {found[:2]}."
        )
        return result

    if kind == "choice":
        expected = spec["expected"]
        # Look for the object label in the final answer.
        labels = re.findall(r"\bobject\s*([ab])\b", lower)
        if not labels:
            labels = re.findall(r"\b([ab])\b", lower)
        actual = labels[-1].upper() if labels else None

        result["passed"] = actual == expected
        result["reason"] = f"Detected choice={actual}; expected Object {expected}."
        return result

    if kind == "uncertainty":
        # Check whether the final answer starts with a clear No.
        # Do not award a pass merely because the explanation contains "not".
        result["passed"] = bool(re.match(r"^\s*no\b", lower))
        result["reason"] = (
            "Expected a clear No because the question asks whether the "
            "claim is established by the given evidence."
        )
        return result

    result["reason"] = f"Unknown scoring type: {kind}"
    return result


def summarize(records, key):
    by_category = defaultdict(list)

    for row in records:
        by_category[row["category"]].append(row[key]["passed"])

    total = len(records)
    passed = sum(row[key]["passed"] for row in records)

    return {
        "passed": passed,
        "total": total,
        "accuracy": passed / total if total else 0.0,
        "by_category": {
            category: {
                "passed": sum(values),
                "total": len(values),
                "accuracy": sum(values) / len(values),
            }
            for category, values in sorted(by_category.items())
        },
    }


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. Rename or inspect it first; "
            "this script will not overwrite results."
        )

    tests = load_tests()
    if len(tests) != 20:
        raise ValueError("This evaluator is configured for 20 test questions.")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_PATH}")
    print(f"Questions: {len(tests)}")
    print("Loading tokenizer and base model...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
    ).to(device)
    base.eval()

    print("Loading LoRA adapter...")
    adapter = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    adapter.eval()

    # Use the same loaded base weights for a fair comparison.
    # Disable the adapter for baseline generation, then enable it.
    records = []

    for index, (test, spec) in enumerate(zip(tests, SPECS), start=1):
        print(f"\n[{index}/{len(tests)}] {test['category']}: {test['question']}")

        adapter.disable_adapter_layers()
        base_response = generate(adapter, tokenizer, device, test["question"])
        base_score = score(base_response, spec)

        adapter.enable_adapter_layers()
        adapter_response = generate(adapter, tokenizer, device, test["question"])
        adapter_score = score(adapter_response, spec)

        record = {
            "index": index,
            "category": test["category"],
            "question": test["question"],
            "reference_answer": test["answer"],
            "scoring_spec": spec,
            "base_output": base_response,
            "adapter_output": adapter_response,
            "base_score": base_score,
            "adapter_score": adapter_score,
        }
        records.append(record)

        print("Base final answer:", base_score["final_answer_used"])
        print("Base score:", base_score["passed"], "-", base_score["reason"])
        print("Adapter final answer:", adapter_score["final_answer_used"])
        print("Adapter score:", adapter_score["passed"], "-", adapter_score["reason"])

    payload = {
        "model": BASE_MODEL,
        "adapter_path": str(ADAPTER_PATH),
        "device": device,
        "evaluation_count": len(records),
        "scoring_note": (
            "Rule-based diagnostic scoring with explicit targets. "
            "Review responses manually; 20 authored questions are not "
            "a statistically robust benchmark."
        ),
        "base_summary": summarize(records, "base_score"),
        "adapter_summary": summarize(records, "adapter_score"),
        "results": records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== V3 SUMMARY =====")
    print(json.dumps({
        "base_summary": payload["base_summary"],
        "adapter_summary": payload["adapter_summary"],
        "output": str(OUTPUT_PATH),
    }, indent=2))
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
