import argparse
import inspect
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


ROOT = Path(__file__).resolve().parents[1]
DPO_PATH = ROOT / "data" / "processed" / "dpo_pairs_sft.jsonl"

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
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_dpo_config(output_dir: Path, args: argparse.Namespace) -> DPOConfig:
    candidate_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-6,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "bf16": True,
        "fp16": False,
        "report_to": "none",
        "remove_unused_columns": False,
        "optim": "adamw_torch",
        "beta": 0.1,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
    }

    signature = inspect.signature(DPOConfig.__init__)
    supported_kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in signature.parameters
    }
    skipped = sorted(set(candidate_kwargs) - set(supported_kwargs))
    if skipped:
        print("Skipped unsupported DPOConfig args:", skipped)

    return DPOConfig(**supported_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="outputs/dpo_smoke")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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
    )
    ref_base.config.use_cache = False
    ref_model = PeftModel.from_pretrained(ref_base, SFT_ADAPTER_DIR, is_trainable=False)
    ref_model.eval()

    rows = load_jsonl(DPO_PATH, limit=args.train_limit)
    dataset = Dataset.from_list(rows)
    print(f"DPO train examples: {len(dataset)}")

    dpo_args = build_dpo_config(output_dir, args)

    trainer_kwargs = {
        "model": model,
        "ref_model": ref_model,
        "args": dpo_args,
        "train_dataset": dataset,
    }

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
