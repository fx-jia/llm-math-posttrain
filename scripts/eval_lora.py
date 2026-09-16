import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.math_verifier import (
    answers_equivalent,
    extract_final_answer,
    has_required_format,
    normalize_answer,
)
from src.prompts import build_math_prompt
from src.run_manifest import write_manifest
TEST_PATH = ROOT / "data" / "processed" / "gsm8k_test.jsonl"
OUTPUT_DIR = ROOT / "outputs"

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=str, required=True)
    parser.add_argument("--data-path", default=str(TEST_PATH.relative_to(ROOT)))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    adapter_dir = ROOT / args.adapter_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    rows = load_jsonl(data_path)
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No evaluation examples selected.")
    set_seed(args.seed)

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
    subset_name = f"limit{args.limit}_seed{args.seed}" if args.limit is not None else "full"
    result_path = OUTPUT_DIR / f"eval_{adapter_name}_{data_path.stem}_{subset_name}.jsonl"
    write_manifest(
        result_path.with_suffix(".manifest.json"),
        ROOT,
        vars(args),
        model_name=MODEL_NAME,
        adapter_dir=str(adapter_dir),
        examples=len(rows),
        decoding="greedy",
        data_path=str(data_path),
    )

    correct = 0
    format_ok = 0
    total_latency = 0.0
    total_output_tokens = 0

    with result_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            prompt = build_math_prompt(row["question"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - start

            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            output = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred = normalize_answer(extract_final_answer(output))
            gold = normalize_answer(row["answer"])

            is_correct = answers_equivalent(output, row["answer"])
            has_format = has_required_format(output)

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
