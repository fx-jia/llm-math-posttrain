"""在 SFT LoRA 适配器上使用偏好对数据继续进行 DPO 训练。"""

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
from trl import DPOConfig, DPOTrainer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.run_manifest import write_manifest
from src.project_config import DEFAULT_BASE_MODEL, DEFAULT_MODEL_REVISION

DPO_PATH = ROOT / "data" / "processed" / "dpo_pairs_sft.jsonl"

MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_BASE_MODEL)
MODEL_REVISION = os.environ.get("MODEL_REVISION", DEFAULT_MODEL_REVISION)
SFT_ADAPTER_DIR = os.environ.get("SFT_ADAPTER_DIR", str(ROOT / "outputs" / "sft_v3_seed42"))


def load_jsonl(path: Path) -> list[dict]:
    """读取 DPO 所需的 prompt/chosen/rejected 三元组。"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append(
                {
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                    "pair_source": item.get("source", "unknown"),
                }
            )
    return rows


def build_dpo_config(output_dir: Path, args: argparse.Namespace) -> tuple[DPOConfig, list[str]]:
    """构建与当前 TRL 版本兼容的 DPO 配置。"""
    candidate_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": args.learning_rate,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "bf16": True,
        "fp16": False,
        "report_to": "none",
        "remove_unused_columns": False,
        "optim": "adamw_torch",
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "seed": args.seed,
        "data_seed": args.seed,
    }

    # TRL 各版本的配置参数有差异，运行时过滤可避免因升降级而报错。
    signature = inspect.signature(DPOConfig.__init__)
    supported_kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in signature.parameters
    }
    skipped = sorted(set(candidate_kwargs) - set(supported_kwargs))
    if skipped:
        print("Skipped unsupported DPOConfig args:", skipped)

    return DPOConfig(**supported_kwargs), skipped


def main() -> None:
    """加载策略/参考模型，并在偏好对上优化 LoRA 参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default=str(DPO_PATH.relative_to(ROOT)))
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="outputs/dpo_smoke")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-legacy-pairs", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_path)
    if not train_path.is_absolute():
        train_path = ROOT / train_path
    rows = load_jsonl(train_path)
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.train_limit is not None:
        rows = rows[: args.train_limit]
    if not rows:
        raise ValueError(f"No DPO pairs selected from {train_path}")
    pair_sources = sorted({row["pair_source"] for row in rows})
    if pair_sources != ["same_policy_verified_candidates"] and not args.allow_legacy_pairs:
        raise ValueError(
            "DPO data is not the V2 same-policy pair format. Rebuild it with "
            "scripts/build_dpo_pairs.py or pass --allow-legacy-pairs. "
            f"Found sources: {pair_sources}"
        )
    dataset = Dataset.from_list(
        [
            {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
            for row in rows
        ]
    )
    print(f"DPO train examples: {len(dataset)}")
    dpo_args, skipped_config_args = build_dpo_config(output_dir, args)

    print(f"Loading tokenizer from SFT adapter: {SFT_ADAPTER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading trainable policy model: {MODEL_NAME} + {SFT_ADAPTER_DIR}")
    policy_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        revision=MODEL_REVISION,
    )
    policy_base.config.use_cache = False
    model = PeftModel.from_pretrained(policy_base, SFT_ADAPTER_DIR, is_trainable=True)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    print(f"Loading frozen reference model: {MODEL_NAME} + {SFT_ADAPTER_DIR}")
    ref_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        revision=MODEL_REVISION,
    )
    ref_base.config.use_cache = False
    # 参考模型固定在 SFT 起点，用于约束策略不要偏离过快。
    ref_model = PeftModel.from_pretrained(ref_base, SFT_ADAPTER_DIR, is_trainable=False)
    ref_model.eval()

    write_manifest(
        output_dir / "run_manifest.json",
        ROOT,
        vars(args),
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        sft_adapter_dir=SFT_ADAPTER_DIR,
        train_path=str(train_path),
        train_examples=len(dataset),
        pair_sources=pair_sources,
        skipped_config_args=skipped_config_args,
    )

    trainer_kwargs = {
        "model": model,
        "ref_model": ref_model,
        "args": dpo_args,
        "train_dataset": dataset,
    }

    # processing_class 是新版 TRL 命名，旧版仍使用 tokenizer。
    trainer_signature = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Saved DPO LoRA adapter and tokenizer to: {output_dir}")
    if torch.cuda.is_available():
        print(f"peak_cuda_memory_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}")


if __name__ == "__main__":
    main()
