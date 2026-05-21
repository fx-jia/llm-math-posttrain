import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.answer_utils import normalize_answer
TEST_PATH = ROOT / "data" / "processed" / "gsm8k_test.jsonl"
OUTPUT_DIR = ROOT / "outputs"

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_final_answer(output: str) -> str:
    match = re.search(r"Final Answer:\s*([^\n]+)", output)
    if match:
        candidate = match.group(1).strip()
    else:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", output.replace(",", ""))
        candidate = numbers[-1] if numbers else ""

    numbers = re.findall(r"-?\d+(?:\.\d+)?", candidate.replace(",", ""))
    if numbers:
        return normalize_answer(numbers[-1])
    return normalize_answer(candidate)


def build_prompt(question: str) -> str:
    return (
        "You are a helpful math reasoning assistant.\n"
        "Solve the following problem step by step, and put the final answer after 'Final Answer:'.\n\n"
        f"Problem:\n{question.strip()}\n\n"
        "Solution:\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    adapter_dir = ROOT / args.adapter_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(TEST_PATH)[: args.limit]

    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    
    adapter_name = adapter_dir.name
    result_path = OUTPUT_DIR / f"eval_{adapter_name}_limit{args.limit}.jsonl"

    correct = 0
    format_ok = 0
    total_latency = 0.0
    total_output_tokens = 0

    with result_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            prompt = build_prompt(row["question"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            start = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            latency = time.time() - start

            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            output = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred = extract_final_answer(output)
            gold = normalize_answer(row["answer"])

            is_correct = pred == gold
            has_format = "Final Answer:" in output

            correct += int(is_correct)
            format_ok += int(has_format)
            total_latency += latency
            total_output_tokens += len(gen_ids)

            record = {
                "id": row["id"],
                "question": row["question"],
                "gold": gold,
                "prediction": pred,
                "correct": is_correct,
                "format_ok": has_format,
                "latency": latency,
                "output_tokens": len(gen_ids),
                "output": output,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"[{idx}/{len(rows)}] correct={is_correct} "
                f"gold={gold} pred={pred} format={has_format} latency={latency:.2f}s"
            )

    n = len(rows)
    print("=" * 80)
    print(f"base_model: {MODEL_NAME}")
    print(f"adapter_dir: {adapter_dir}")
    print(f"examples: {n}")
    print(f"exact_match: {correct / n:.4f}")
    print(f"format_rate: {format_ok / n:.4f}")
    print(f"avg_latency_sec: {total_latency / n:.2f}")
    print(f"avg_output_tokens: {total_output_tokens / n:.1f}")
    if torch.cuda.is_available():
        print(f"peak_cuda_memory_gb: {torch.cuda.max_memory_allocated() / 1024**3:.2f}")
    print(f"saved results: {result_path}")


if __name__ == "__main__":
    main()
