"""Sampling evaluation with pass@k, majority vote, and token-efficiency metrics."""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.math_verifier import answers_equivalent, normalize_answer
from src.project_config import DEFAULT_BASE_MODEL
from src.prompts import build_math_prompt, render_prompt_for_model
from src.run_manifest import write_manifest
from src.statistics import estimate_pass_at_k


TEST_PATH = ROOT / "data" / "processed" / "gsm8k_test.jsonl"
OUTPUT_DIR = ROOT / "outputs"
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_BASE_MODEL)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir")
    parser.add_argument("--data-path", default=str(TEST_PATH.relative_to(ROOT)))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--pass-k", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    invalid_k = [k for k in args.pass_k if k < 1 or k > args.num_samples]
    if invalid_k:
        raise ValueError(f"pass@k values must be within [1, {args.num_samples}]: {invalid_k}")

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    rows = load_jsonl(data_path)
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit] if args.limit is not None else rows
    set_seed(args.seed)

    adapter_dir = None
    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir)
        if not adapter_dir.is_absolute():
            adapter_dir = ROOT / adapter_dir

    tokenizer_source = adapter_dir or MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir) if adapter_dir else base_model
    model.eval()

    model_label = adapter_dir.name if adapter_dir else "base"
    subset = f"limit{len(rows)}_seed{args.seed}"
    result_path = OUTPUT_DIR / f"eval_sampling_{model_label}_{data_path.stem}_{subset}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(
        result_path.with_suffix(".manifest.json"),
        ROOT,
        vars(args),
        model_name=MODEL_NAME,
        adapter_dir=str(adapter_dir) if adapter_dir else None,
        examples=len(rows),
        decoding="sampling",
        data_path=str(data_path),
    )

    pass_totals = {k: 0.0 for k in args.pass_k}
    majority_correct = 0
    total_tokens = 0
    total_latency = 0.0

    with result_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            prompt = render_prompt_for_model(
                tokenizer,
                build_math_prompt(row["question"]),
                args.use_chat_template,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                sequences = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.num_samples,
                    pad_token_id=tokenizer.eos_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - start

            prompt_length = inputs["input_ids"].shape[1]
            candidates = []
            for sequence in sequences:
                generated = sequence[prompt_length:]
                token_ids = generated.tolist()
                if tokenizer.eos_token_id in token_ids:
                    token_count = token_ids.index(tokenizer.eos_token_id) + 1
                else:
                    token_count = len(token_ids)
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                prediction = normalize_answer(text)
                candidates.append(
                    {
                        "prediction": prediction,
                        "correct": answers_equivalent(text, row["answer"]),
                        "tokens": token_count,
                        "output": text,
                    }
                )

            correct_count = sum(item["correct"] for item in candidates)
            pass_scores = {
                str(k): estimate_pass_at_k(args.num_samples, correct_count, k)
                for k in args.pass_k
            }
            for k in args.pass_k:
                pass_totals[k] += pass_scores[str(k)]

            valid_predictions = [item["prediction"] for item in candidates if item["prediction"]]
            majority_prediction = Counter(valid_predictions).most_common(1)[0][0] if valid_predictions else ""
            is_majority_correct = answers_equivalent(majority_prediction, row["answer"])
            majority_correct += int(is_majority_correct)
            total_tokens += sum(item["tokens"] for item in candidates)
            total_latency += latency

            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "gold": normalize_answer(row["answer"]),
                        "num_correct": correct_count,
                        "pass_at_k": pass_scores,
                        "majority_prediction": majority_prediction,
                        "majority_correct": is_majority_correct,
                        "latency": latency,
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"[{index}/{len(rows)}] correct_samples={correct_count}/{args.num_samples}")

    print(f"model: {model_label}")
    print(f"examples: {len(rows)}")
    for k in args.pass_k:
        print(f"pass@{k}: {pass_totals[k] / len(rows):.4f}")
    print(f"majority@{args.num_samples}: {majority_correct / len(rows):.4f}")
    print(f"avg_tokens_per_problem: {total_tokens / len(rows):.1f}")
    print(f"avg_latency_sec: {total_latency / len(rows):.2f}")
    print(f"saved results: {result_path}")


if __name__ == "__main__":
    main()
