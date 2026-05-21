import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rewards import correctness_reward, format_reward, length_reward


GRPO_PATH = ROOT / "data" / "processed" / "grpo_train.jsonl"

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
SFT_ADAPTER_DIR = os.environ.get("SFT_ADAPTER_DIR", str(ROOT / "outputs" / "sft_lora_r8_full"))


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append(
                {
                    "prompt": item["prompt"],
                    "answer": item["answer"],
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_grpo_config(output_dir: Path, args: argparse.Namespace) -> GRPOConfig:
    candidate_kwargs = {
        "output_dir": str(output_dir),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": args.max_steps,
        "bf16": True,
        "fp16": False,
        "report_to": "none",
        "remove_unused_columns": False,
        "optim": "adamw_torch",
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "beta": args.beta,
    }

    signature = inspect.signature(GRPOConfig.__init__)
    supported_kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in signature.parameters
    }
    skipped = sorted(set(candidate_kwargs) - set(supported_kwargs))
    if skipped:
        print("Skipped unsupported GRPOConfig args:", skipped)

    return GRPOConfig(**supported_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="outputs/grpo_smoke")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.01)
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from SFT adapter: {SFT_ADAPTER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading policy model: {MODEL_NAME} + {SFT_ADAPTER_DIR}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_DIR, is_trainable=True)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    rows = load_jsonl(GRPO_PATH, limit=args.train_limit)
    dataset = Dataset.from_list(rows)
    print(f"GRPO train examples: {len(dataset)}")
    print(f"num_generations: {args.num_generations}")

    grpo_args = build_grpo_config(output_dir, args)

    trainer_kwargs = {
        "model": model,
        "reward_funcs": [correctness_reward, format_reward, length_reward],
        "args": grpo_args,
        "train_dataset": dataset,
    }

    trainer_signature = inspect.signature(GRPOTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Saved GRPO LoRA adapter and tokenizer to: {output_dir}")
    if torch.cuda.is_available():
        print(f"peak_cuda_memory_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}")


if __name__ == "__main__":
    main()
