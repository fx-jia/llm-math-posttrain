"""Evaluate a larger OpenAI-compatible API model with the shared verifier."""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.math_verifier import answers_equivalent, normalize_answer
from src.project_config import DEFAULT_API_BASE_URL, DEFAULT_API_MODEL
from src.prompts import build_math_prompt
from src.run_manifest import write_manifest
from src.statistics import estimate_pass_at_k, wilson_interval


DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "hmmt_feb_2026.jsonl"
OUTPUT_DIR = ROOT / "outputs"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _usage_dict(usage) -> dict:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _request_candidate(client, args, prompt: str, seed: int) -> dict:
    start = time.perf_counter()
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                seed=seed,
                n=1,
                extra_body={"top_k": args.top_k},
            )
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            reasoning = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or ""
            )
            output = f"{reasoning}\n{content}".strip() if reasoning else content.strip()
            return {
                "output": output,
                "content": content,
                "reasoning": reasoning,
                "finish_reason": getattr(choice, "finish_reason", None),
                "latency": time.perf_counter() - start,
                "usage": _usage_dict(response.usage),
                "error": None,
            }
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < args.retries:
                time.sleep(min(2**attempt, 4))
    return {
        "output": "",
        "content": "",
        "reasoning": "",
        "finish_reason": None,
        "latency": time.perf_counter() - start,
        "usage": _usage_dict(None),
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("API_MODEL_NAME", DEFAULT_API_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL))
    parser.add_argument("--api-key-env", default="TOGETHER_API_KEY")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH.relative_to(ROOT)))
    parser.add_argument("--output-path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before running API evaluation.")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")

    from openai import OpenAI

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    rows = _load_jsonl(data_path)
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No API evaluation examples selected.")

    model_slug = args.model.replace("/", "_").replace(".", "_")
    output_path = (
        Path(args.output_path)
        if args.output_path
        else OUTPUT_DIR / f"eval_api_{model_slug}_{data_path.stem}_seed{args.seed}.jsonl"
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    pass1_total = 0.0
    passk_total = 0.0
    majority_total = 0
    problem_any_correct = 0
    failed_requests = 0
    completion_tokens = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for problem_index, row in enumerate(rows, start=1):
            prompt = build_math_prompt(row["question"])
            candidates = []
            for sample_index in range(args.num_samples):
                candidate = _request_candidate(
                    client,
                    args,
                    prompt,
                    seed=args.seed + problem_index * 1000 + sample_index,
                )
                candidate["prediction"] = normalize_answer(candidate["output"])
                candidate["correct"] = bool(candidate["output"]) and answers_equivalent(
                    candidate["output"], row["answer"]
                )
                failed_requests += int(candidate["error"] is not None)
                completion_tokens += candidate["usage"]["completion_tokens"]
                candidates.append(candidate)

            correct_count = sum(candidate["correct"] for candidate in candidates)
            pass1 = correct_count / args.num_samples
            passk = estimate_pass_at_k(args.num_samples, correct_count, args.num_samples)
            predictions = [candidate["prediction"] for candidate in candidates if candidate["prediction"]]
            majority_prediction = Counter(predictions).most_common(1)[0][0] if predictions else ""
            majority_correct = bool(majority_prediction) and answers_equivalent(
                majority_prediction, row["answer"]
            )
            pass1_total += pass1
            passk_total += passk
            majority_total += int(majority_correct)
            problem_any_correct += int(correct_count > 0)

            handle.write(
                json.dumps(
                    {
                        "id": row.get("id", problem_index - 1),
                        "benchmark": row.get("benchmark", data_path.stem),
                        "question": row["question"],
                        "gold": row["answer"],
                        "num_correct": correct_count,
                        "pass_at_1": pass1,
                        f"pass_at_{args.num_samples}": passk,
                        "majority_prediction": majority_prediction,
                        "majority_correct": majority_correct,
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"[{problem_index}/{len(rows)}] correct={correct_count}/{args.num_samples}")

    low, high = wilson_interval(problem_any_correct, len(rows))
    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "examples": len(rows),
        "samples_per_problem": args.num_samples,
        "mean_sample_accuracy": pass1_total / len(rows),
        f"pass_at_{args.num_samples}": passk_total / len(rows),
        f"empirical_any_correct_at_{args.num_samples}": problem_any_correct / len(rows),
        "any_correct_wilson_95": [low, high],
        f"majority_at_{args.num_samples}": majority_total / len(rows),
        "completion_tokens": completion_tokens,
        "completion_tokens_per_problem": completion_tokens / len(rows),
        "failed_requests": failed_requests,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    safe_arguments = {**vars(args), "api_key_env": args.api_key_env}
    write_manifest(
        output_path.with_suffix(".manifest.json"),
        ROOT,
        safe_arguments,
        model_name=args.model,
        provider_base_url=args.base_url,
        data_path=str(data_path),
        examples=len(rows),
        decoding="sampling",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
