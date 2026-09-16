"""使用数学推理轨迹对因果语言模型进行 response-only LoRA SFT。"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.run_manifest import write_manifest
from src.project_config import DEFAULT_BASE_MODEL, DEFAULT_MODEL_REVISION, lora_target_modules
from src.sft_data import encode_response_only

TRAIN_PATH = ROOT / "data" / "processed" / "openr1_sft_v3.jsonl"
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_BASE_MODEL)
MODEL_REVISION = os.environ.get("MODEL_REVISION", DEFAULT_MODEL_REVISION)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class ResponseOnlyCollator:
    """Right-pad causal LM inputs while preserving ``-100`` prompt labels."""

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_length)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    """构建 LoRA 模型、分词数据并启动 SFT 训练。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default=str(TRAIN_PATH.relative_to(ROOT)))
    parser.add_argument(
        "--init-adapter",
        help="Optional LoRA adapter to continue training (used by the RFT baseline).",
    )
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="outputs/sft_v3_smoke")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    init_adapter = None
    if args.init_adapter:
        init_adapter = Path(args.init_adapter)
        if not init_adapter.is_absolute():
            init_adapter = ROOT / init_adapter
    tokenizer_source = init_adapter or MODEL_NAME
    print(f"Loading tokenizer: {tokenizer_source}")
    tokenizer_kwargs = {"trust_remote_code": True}
    if not init_adapter:
        tokenizer_kwargs["revision"] = MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        revision=MODEL_REVISION,
    )
    model.config.use_cache = False

    if init_adapter:
        print(f"Continuing trainable LoRA adapter: {init_adapter}")
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        target_modules = lora_target_modules(MODEL_NAME)
        print(f"LoRA target modules: {target_modules}")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    train_path = Path(args.train_path)
    if not train_path.is_absolute():
        train_path = ROOT / train_path
    rows = load_jsonl(train_path)
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.train_limit is not None:
        rows = rows[: args.train_limit]

    encoded, dropped = [], 0
    for row in rows:
        item = encode_response_only(
            row,
            tokenizer,
            args.max_length,
            use_chat_template=args.use_chat_template,
        )
        if item is None:
            dropped += 1
        else:
            encoded.append(item)
    if not encoded:
        raise ValueError("No SFT examples remain after max-length filtering.")
    tokenized = Dataset.from_list(encoded)
    collator = ResponseOnlyCollator(tokenizer)
    print(
        f"SFT examples: kept={len(encoded)}, dropped_overlength={dropped}, "
        f"max_length={args.max_length}, seed={args.seed}"
    )

    write_manifest(
        output_dir / "run_manifest.json",
        ROOT,
        vars(args),
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        train_examples=len(encoded),
        dropped_overlength=dropped,
        objective="response_only_causal_lm",
        train_path=str(train_path),
        lora_target_modules=lora_target_modules(MODEL_NAME),
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=args.seed,
        data_seed=args.seed,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Saved LoRA adapter and tokenizer to: {output_dir}")
    if torch.cuda.is_available():
        print(f"peak_cuda_memory_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}")


if __name__ == "__main__":
    main()
