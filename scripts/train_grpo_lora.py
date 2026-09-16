"""从 SFT LoRA 起点出发，使用可验证数学奖励进行 GRPO 训练。"""

import argparse
import inspect
import json
import os
import random
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
from src.run_manifest import write_manifest


GRPO_PATH = ROOT / "data" / "processed" / "grpo_train.jsonl"
FRONTIER_PATH = ROOT / "data" / "processed" / "rlvr_frontier_sft.jsonl"

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
SFT_ADAPTER_DIR = os.environ.get("SFT_ADAPTER_DIR", str(ROOT / "outputs" / "sft_lora_r8_full"))


def load_jsonl(path: Path) -> list[dict]:
    """读取 GRPO 需要的题目 prompt 和标准答案。"""
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
    return rows


def build_grpo_config(output_dir: Path, args: argparse.Namespace) -> tuple[GRPOConfig, list[str]]:
    """Build a stable DAPO-style config and reject silent recipe downgrades."""
    scale_rewards = False if args.scale_rewards == "none" else args.scale_rewards
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
        "loss_type": args.loss_type,
        "scale_rewards": scale_rewards,
        "epsilon": args.epsilon,
        "epsilon_high": args.epsilon_high,
        "mask_truncated_completions": args.mask_truncated_completions,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "data_seed": args.seed,
    }

    # 通过反射兼容 TRL API 变化，同时打印被忽略项以防静默失配。
    signature = inspect.signature(GRPOConfig.__init__)
    supported_kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in signature.parameters
    }
    skipped = sorted(set(candidate_kwargs) - set(supported_kwargs))
    if skipped:
        print("Skipped unsupported GRPOConfig args:", skipped)

    stability_args = {
        "loss_type",
        "scale_rewards",
        "epsilon_high",
        "mask_truncated_completions",
    }
    missing_stability_args = sorted(stability_args & set(skipped))
    if missing_stability_args and not args.allow_legacy_trl:
        raise RuntimeError(
            "The installed TRL cannot run the requested stable RLVR recipe. "
            f"Unsupported arguments: {missing_stability_args}. Upgrade TRL or pass "
            "--allow-legacy-trl to accept a documented downgrade."
        )

    return GRPOConfig(**supported_kwargs), skipped


def main() -> None:
    """构建可训练策略模型，按组采样回答并用规则奖励优化。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default=str(FRONTIER_PATH.relative_to(ROOT)))
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="outputs/rlvr_dapo")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--loss-type", choices=["grpo", "bnpo", "dr_grpo", "dapo"], default="dapo")
    parser.add_argument("--scale-rewards", choices=["none", "group", "batch"], default="none")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-high", type=float, default=0.28)
    parser.add_argument(
        "--mask-truncated-completions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--reward-mode", choices=["correctness", "shaped"], default="correctness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-legacy-trl", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_path)
    if not train_path.is_absolute():
        train_path = ROOT / train_path
    if not train_path.exists() and train_path == FRONTIER_PATH:
        print(
            f"Frontier dataset not found at {train_path}; falling back to {GRPO_PATH}. "
            "Run build_dpo_pairs.py first for difficulty-aware RLVR."
        )
        train_path = GRPO_PATH
    rows = load_jsonl(train_path)
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.train_limit is not None:
        rows = rows[: args.train_limit]
    if not rows:
        raise ValueError(f"No GRPO examples selected from {train_path}")
    dataset = Dataset.from_list(rows)
    print(f"GRPO train examples: {len(dataset)}")
    print(f"num_generations: {args.num_generations}")
    print(f"train_path: {train_path}")
    grpo_args, skipped_config_args = build_grpo_config(output_dir, args)

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

    reward_funcs = [correctness_reward]
    if args.reward_mode == "shaped":
        reward_funcs.extend([format_reward, length_reward])

    write_manifest(
        output_dir / "run_manifest.json",
        ROOT,
        vars(args),
        model_name=MODEL_NAME,
        sft_adapter_dir=SFT_ADAPTER_DIR,
        train_path=str(train_path),
        train_examples=len(dataset),
        reward_functions=[function.__name__ for function in reward_funcs],
        skipped_config_args=skipped_config_args,
    )

    # Correctness-only is the default so formatting cannot masquerade as reasoning progress.
    trainer_kwargs = {
        "model": model,
        "reward_funcs": reward_funcs,
        "args": grpo_args,
        "train_dataset": dataset,
    }

    # 同时支持新版 processing_class 和旧版 tokenizer 接口。
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
