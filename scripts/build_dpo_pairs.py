"""Build policy-matched DPO pairs and an RLVR difficulty-frontier dataset."""

import argparse
import json
import os
import random
import sys
import time
from itertools import product
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.math_verifier import answers_equivalent, extract_final_answer, has_required_format
from src.project_config import DEFAULT_BASE_MODEL
from src.prompts import build_math_prompt, render_prompt_for_model
from src.run_manifest import write_manifest


FULL_TRAIN_PATH = ROOT / "data" / "processed" / "gsm8k_train.jsonl"
CORE_TRAIN_PATH = ROOT / "data" / "processed" / "gsm8k_train_core.jsonl"
TRAIN_PATH = CORE_TRAIN_PATH if CORE_TRAIN_PATH.exists() else FULL_TRAIN_PATH
DEFAULT_PAIR_PATH = ROOT / "data" / "processed" / "dpo_pairs_sft.jsonl"
DEFAULT_FRONTIER_PATH = ROOT / "data" / "processed" / "rlvr_frontier_sft.jsonl"
DEFAULT_STATS_PATH = ROOT / "data" / "processed" / "rollout_stats_sft.jsonl"
DEFAULT_RFT_PATH = ROOT / "data" / "processed" / "rft_correct_sft.jsonl"

MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_BASE_MODEL)
ADAPTER_DIR = os.environ.get("SFT_ADAPTER_DIR", str(ROOT / "outputs" / "sft_lora_r8_full"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def candidate_record(text: str, token_count: int, gold: str) -> dict:
    return {
        "text": text,
        "prediction": extract_final_answer(text),
        "correct": answers_equivalent(text, gold),
        "format_ok": has_required_format(text),
        "tokens": token_count,
    }


def closest_length_pair(correct: list[dict], incorrect: list[dict]) -> tuple[dict, dict]:
    """Remove the easiest DPO shortcut by matching response lengths."""
    return min(
        product(correct, incorrect),
        key=lambda pair: (abs(pair[0]["tokens"] - pair[1]["tokens"]), pair[0]["tokens"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default=str(TRAIN_PATH.relative_to(ROOT)))
    parser.add_argument("--input-limit", type=int, default=800)
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-format", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pairs-output", default=str(DEFAULT_PAIR_PATH.relative_to(ROOT)))
    parser.add_argument("--frontier-output", default=str(DEFAULT_FRONTIER_PATH.relative_to(ROOT)))
    parser.add_argument("--stats-output", default=str(DEFAULT_STATS_PATH.relative_to(ROOT)))
    parser.add_argument("--rft-output", default=str(DEFAULT_RFT_PATH.relative_to(ROOT)))
    args = parser.parse_args()

    if args.num_candidates < 2:
        raise ValueError("--num-candidates must be at least 2 to estimate a preference frontier.")

    pair_path = resolve_path(args.pairs_output)
    frontier_path = resolve_path(args.frontier_output)
    stats_path = resolve_path(args.stats_output)
    rft_path = resolve_path(args.rft_output)
    for path in (pair_path, frontier_path, stats_path, rft_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading base model: {MODEL_NAME}")
    print(f"Loading SFT adapter: {ADAPTER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    train_path = resolve_path(args.train_path)
    rows = load_jsonl(train_path)
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.input_limit]

    pair_count = 0
    frontier_count = 0
    processed = 0
    rft_count = 0
    start_time = time.time()

    with (
        pair_path.open("w", encoding="utf-8") as pair_file,
        frontier_path.open("w", encoding="utf-8") as frontier_file,
        stats_path.open("w", encoding="utf-8") as stats_file,
        rft_path.open("w", encoding="utf-8") as rft_file,
    ):
        for processed, row in enumerate(rows, start=1):
            question = row["question"]
            prompt = render_prompt_for_model(
                tokenizer,
                build_math_prompt(question),
                args.use_chat_template,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.num_candidates,
                    pad_token_id=tokenizer.eos_token_id,
                )

            prompt_length = inputs["input_ids"].shape[1]
            candidates = []
            for sequence in output_ids:
                generated = sequence[prompt_length:]
                token_ids = generated.tolist()
                if tokenizer.eos_token_id in token_ids:
                    token_count = token_ids.index(tokenizer.eos_token_id) + 1
                else:
                    token_count = len(token_ids)
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                candidates.append(candidate_record(text, token_count, row["answer"]))

            correct = [item for item in candidates if item["correct"]]
            incorrect = [item for item in candidates if not item["correct"]]
            formatted_correct = [item for item in correct if item["format_ok"]]
            formatted_incorrect = [item for item in incorrect if item["format_ok"]]
            pass_rate = len(correct) / len(candidates)

            stats_file.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "num_candidates": len(candidates),
                        "num_correct": len(correct),
                        "num_formatted": sum(item["format_ok"] for item in candidates),
                        "pass_rate": pass_rate,
                        "mean_tokens": sum(item["tokens"] for item in candidates) / len(candidates),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            # Keep prompts where the current policy produces both outcomes.
            if correct and incorrect:
                frontier_file.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "question": question,
                            "prompt": prompt,
                            "answer": row["answer"],
                            "sft_pass_rate": pass_rate,
                            "num_candidates": len(candidates),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                frontier_count += 1

            positive_pool = formatted_correct if args.require_format else correct
            negative_pool = formatted_incorrect if args.require_format else incorrect
            if positive_pool:
                rft_choice = positive_pool[0]
                rft_file.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "source": "same_policy_verified_correct",
                            "question": question,
                            "prompt": prompt,
                            "completion": rft_choice["text"],
                            "answer": row["answer"],
                            "sft_pass_rate": pass_rate,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rft_count += 1

            if positive_pool and negative_pool and pair_count < args.max_pairs:
                chosen, rejected = closest_length_pair(positive_pool, negative_pool)
                pair_file.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "source": "same_policy_verified_candidates",
                            "question": question,
                            "prompt": prompt,
                            "chosen": chosen["text"],
                            "rejected": rejected["text"],
                            "gold": row["answer"],
                            "chosen_prediction": chosen["prediction"],
                            "rejected_prediction": rejected["prediction"],
                            "chosen_tokens": chosen["tokens"],
                            "rejected_tokens": rejected["tokens"],
                            "length_gap": abs(chosen["tokens"] - rejected["tokens"]),
                            "sft_pass_rate": pass_rate,
                            "num_candidates": len(candidates),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                pair_count += 1

            if processed % 25 == 0:
                print(
                    f"processed={processed}, pairs={pair_count}, "
                    f"frontier_prompts={frontier_count}"
                )
            if pair_count >= args.max_pairs:
                break

    elapsed = time.time() - start_time
    write_manifest(
        pair_path.with_suffix(".manifest.json"),
        ROOT,
        vars(args),
        model_name=MODEL_NAME,
        adapter_dir=ADAPTER_DIR,
        train_path=str(train_path),
        processed_examples=processed,
        saved_pairs=pair_count,
        frontier_prompts=frontier_count,
        rft_examples=rft_count,
    )

    print(f"processed examples: {processed}")
    print(f"saved policy-matched pairs: {pair_count} -> {pair_path}")
    print(f"saved frontier prompts: {frontier_count} -> {frontier_path}")
    print(f"saved rollout statistics: {stats_path}")
    print(f"saved verified RFT examples: {rft_count} -> {rft_path}")
    print(f"elapsed seconds: {elapsed:.2f}")
    if torch.cuda.is_available():
        print(f"peak_cuda_memory_gb: {torch.cuda.max_memory_allocated() / 1024**3:.2f}")


if __name__ == "__main__":
    main()
