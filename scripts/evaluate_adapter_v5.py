import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/reasoning_eval_v1.jsonl"
ADAPTER_PATH = ROOT / "models/final_adapter"
OUTPUT_PATH = ROOT / "results/base_vs_adapter_v5.json"
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
        raise ValueError(f"Expected {len(SPECS)} questions, got {len(tests)}")
    return tests


def make_prompt(tokenizer, question):
    messages = [
        {
            "role": "system",
            "content": (
                "Answer accurately and concisely. Do not show extended reasoning. "
                "Start with ANSWER: followed by the answer. "
                "For object comparisons, answer Object A or Object B. "
                "For yes/no questions, answer Yes or No."
            ),
        },
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def generate(model, tokenizer, device, question):
    inputs = tokenizer(
        make_prompt(tokenizer, question), return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )

    answer_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()


def extract_answer(response):
    matches = re.findall(
        r"(?im)^\s*ANSWER\s*:\s*(.+?)\s*$", response
    )
    return matches[-1].strip() if matches else ""


def numbers(text):
    return [
        float(x.replace(",", ""))
        for x in re.findall(
            r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text
        )
    ]


def score(response, spec):
    answer = extract_answer(response)
    found = numbers(answer)
    lower = answer.lower()
    kind = spec["type"]
    passed = False

    if kind in ("numeric", "signed_direction", "numeric_direction"):
        expected = spec["expected"]
        numeric_ok = bool(found) and abs(
            found[0] - expected
        ) <= spec["tolerance"]
        direction_ok = (
            kind == "numeric"
            or (
                kind == "signed_direction" and "left" in lower
            )
            or (
                kind == "numeric_direction" and "right" in lower
            )
        )
        passed = numeric_ok and direction_ok
        reason = (
            f"Expected {expected}; extracted numbers={found}; "
            f"direction_ok={direction_ok}"
        )

    elif kind == "two_numeric":
        expected = spec["expected"]
        passed = (
            len(found) >= 2
            and all(
                abs(found[i] - target) <= spec["tolerance"]
                for i, target in enumerate(expected)
            )
        )
        reason = f"Expected {expected}; extracted numbers={found}"

    elif kind == "choice":
        match = re.search(r"\bobject\s+([ab])\b", lower)
        actual = match.group(1).upper() if match else None
        passed = actual == spec["expected"]
        reason = f"Expected Object {spec['expected']}; got {actual}"

    elif kind == "uncertainty":
        passed = bool(re.match(r"^no\b", answer, re.IGNORECASE))
        reason = "Expected a direct No answer"

    else:
        reason = f"Unknown scoring type: {kind}"

    if not answer:
        reason = "Missing ANSWER line; response cannot be scored reliably"

    return {
        "passed": passed,
        "response": response,
        "answer_extracted": answer,
        "numbers_found": found,
        "expected": spec,
        "reason": reason,
    }


def summarize(records, key):
    total = len(records)
    passed = sum(int(r[key]["passed"]) for r in records)
    categories = {}

    for r in records:
        bucket = categories.setdefault(
            r["category"], {"passed": 0, "total": 0}
        )
        bucket["total"] += 1
        bucket["passed"] += int(r[key]["passed"])

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
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT_PATH}"
        )

    tests = load_tests()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)
    print("Questions:", len(tests))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32
    ).to(device)
    base.eval()

    adapted = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    adapted.eval()

    sanity = (
        "An object's center moves by dx=3 pixels and dy=4 pixels. "
        "What is its displacement magnitude?"
    )
    print("\nSANITY CHECK")
    print("Base:", repr(generate(adapted, tokenizer, device, sanity)[:300]))
    adapted.disable_adapter_layers()
    print("Base with adapter disabled:", "ready")
    adapted.enable_adapter_layers()
    print("Adapter:", repr(generate(adapted, tokenizer, device, sanity)[:300]))

    records = []
    for i, (test, spec) in enumerate(zip(tests, SPECS), start=1):
        print(f"\n[{i}/{len(tests)}] {test['category']}: {test['question']}")

        adapted.disable_adapter_layers()
        base_response = generate(adapted, tokenizer, device, test["question"])
        adapted.enable_adapter_layers()
        adapter_response = generate(adapted, tokenizer, device, test["question"])

        base_score = score(base_response, spec)
        adapter_score = score(adapter_response, spec)

        records.append({
            "index": i,
            "category": test["category"],
            "question": test["question"],
            "reference_answer": test["answer"],
            "base_output": base_response,
            "adapter_output": adapter_response,
            "base_score": base_score,
            "adapter_score": adapter_score,
        })

        print("Base answer:", repr(base_score["answer_extracted"]))
        print("Base pass:", base_score["passed"], base_score["reason"])
        print("Adapter answer:", repr(adapter_score["answer_extracted"]))
        print("Adapter pass:", adapter_score["passed"], adapter_score["reason"])

    payload = {
        "model": BASE_MODEL,
        "adapter_path": str(ADAPTER_PATH),
        "device": device,
        "evaluation_count": len(records),
        "note": (
            "Small authored diagnostic set, not a robust benchmark. "
            "Inspect responses manually; scores alone do not prove "
            "general reasoning ability or research-paper reproduction."
        ),
        "base_summary": summarize(records, "base_score"),
        "adapter_summary": summarize(records, "adapter_score"),
        "results": records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nV5 SUMMARY")
    print(json.dumps({
        "base_summary": payload["base_summary"],
        "adapter_summary": payload["adapter_summary"],
        "output": str(OUTPUT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
