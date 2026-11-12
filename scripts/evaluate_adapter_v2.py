import gc
import json
import math
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3-0.6B"
ADAPTER = Path("models/final_adapter")
DATA = Path("data/reasoning_eval_v1.jsonl")
OUT = Path("results/base_vs_adapter_v2.json")


def load_tests():
    if not DATA.exists():
        raise FileNotFoundError(f"Evaluation data not found: {DATA}")

    tests = [
        json.loads(line)
        for line in DATA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not tests:
        raise ValueError("Evaluation dataset is empty.")

    required = {"category", "question", "answer"}
    for i, item in enumerate(tests, start=1):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Example {i} missing fields: {missing}")

    if len({x["question"] for x in tests}) != len(tests):
        raise ValueError("Duplicate evaluation questions detected.")

    return tests


def generate(model, tokenizer, device, question):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant for computer vision research. "
                "Answer the question using only the supplied information. "
                "Show calculations when useful. If information is missing, "
                "say so rather than inventing observations. Be concise."
            ),
        },
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(
        new_tokens, skip_special_tokens=True
    ).strip()


def numbers(text):
    return [
        float(x)
        for x in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text)
    ]


def score_answer(test, response):
    """Conservative, category-specific checks; not semantic judging."""
    category = test["category"]
    answer = test["answer"].lower()
    text = response.lower()
    expected_nums = numbers(test["answer"])
    found_nums = numbers(response)

    if category == "motion_calculation":
        if not expected_nums:
            return None
        target = expected_nums[-1]
        return any(abs(x - target) <= 0.05 for x in found_nums)

    if category == "spatial_reasoning":
        if "object a" in answer and "object b" not in answer:
            return "object a" in text and "object b" not in text
        if "object b" in answer and "object a" not in answer:
            return "object b" in text and "object a" not in text
        if "object a" in answer and "object b" in answer:
            # Area questions: the expected larger object is stated first.
            expected = re.search(r"(object [ab]).{0,50}(?:area|larger)", answer)
            if expected:
                return expected.group(1) in text
            return None
        return None

    if category == "segmentation_metrics":
        if not expected_nums:
            return None
        target = expected_nums[-1]
        return any(abs(x - target) <= 0.015 for x in found_nums)

    if category == "temporal_reasoning":
        if not expected_nums:
            return None
        target = expected_nums[-1]
        return any(abs(x - target) <= 0.05 for x in found_nums)

    if category == "uncertainty":
        if any(term in answer for term in (
            "no.", "not ", "insufficient", "cannot", "does not"
        )):
            return any(term in text for term in (
                "cannot", "can't", "not enough", "insufficient",
                "does not", "doesn't", "no,", "no.", "not possible",
                "not prove", "not establish", "not the same"
            ))
        return None

    return None


def main():
    tests = load_tests()

    if not ADAPTER.is_dir():
        raise FileNotFoundError(f"Adapter directory missing: {ADAPTER}")

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 64)
    print("BASE MODEL vs LORA ADAPTER — EVALUATION V2")
    print("=" * 64)
    print("Device:", device)
    print("Tests:", len(tests))
    print("Dataset:", DATA)
    print("Adapter:", ADAPTER)
    print("Output:", OUT)
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

    for i, test in enumerate(tests, start=1):
        print(f"\n[{i}/{len(tests)}] BASE | {test['category']}")
        start = time.time()
        response = generate(base, tokenizer, device, test["question"])

        results.append({
            **test,
            "base_output": response,
            "base_seconds": round(time.time() - start, 3),
        })
        print("Question:", test["question"])
        print("Expected:", test["answer"])
        print("Base output:", response[:500])

    print("\nLoading LoRA adapter...")
    adapted = PeftModel.from_pretrained(
        base, str(ADAPTER), is_trainable=False
    )
    adapted.eval()

    for i, result in enumerate(results, start=1):
        print(f"\n[{i}/{len(results)}] ADAPTER | {result['category']}")
        start = time.time()
        response = generate(
            adapted, tokenizer, device, result["question"]
        )
        result["adapter_output"] = response
        result["adapter_seconds"] = round(time.time() - start, 3)

        print("Expected:", result["answer"])
        print("Adapter output:", response[:500])

    for result in results:
        result["base_rule_score"] = score_answer(
            result, result["base_output"]
        )
        result["adapter_rule_score"] = score_answer(
            result, result["adapter_output"]
        )

    def summarize(score_key):
        scored = [r for r in results if r[score_key] is not None]
        correct = sum(r[score_key] is True for r in scored)
        return {
            "scored": len(scored),
            "unscored": len(results) - len(scored),
            "rule_check_passed": correct,
            "rule_check_accuracy": (
                round(correct / len(scored), 4) if scored else None
            ),
        }

    report = {
        "base_model": BASE,
        "adapter_path": str(ADAPTER),
        "dataset": str(DATA),
        "device": device,
        "test_count": len(results),
        "base_summary": summarize("base_rule_score"),
        "adapter_summary": summarize("adapter_rule_score"),
        "scoring_warning": (
            "Rule-based diagnostic only. Numeric and keyword checks may "
            "misclassify valid answers. Manually review outputs before "
            "claiming that the adapter improves reasoning."
        ),
        "results": results,
    }

    # Do not overwrite previous V2 results.
    if OUT.exists():
        raise FileExistsError(
            f"{OUT} already exists; refusing to overwrite it."
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 64)
    print("EVALUATION COMPLETE")
    print("Saved:", OUT)
    print("Base summary:", json.dumps(report["base_summary"]))
    print("Adapter summary:", json.dumps(report["adapter_summary"]))
    print("Review individual answers before drawing conclusions.")
    print("=" * 64)

    del adapted, base
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
