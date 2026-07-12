"""使用 GSM8K 推理文本对因果语言模型进行 LoRA 监督微调。"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "gsm8k_train.jsonl"
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """按顺序读取 JSONL；limit 用于快速烟雾测试。"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def main() -> None:
    """构建 LoRA 模型、分词数据并启动 SFT 训练。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="outputs/sft_smoke")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 同时适配注意力与 MLP 线性层，以少量参数覆盖主要变换路径。
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    rows = load_jsonl(TRAIN_PATH, limit=args.train_limit)
    dataset = Dataset.from_list([{"text": row["sft_text"]} for row in rows])

    def tokenize_fn(batch):
        """仅做截断；动态 padding 交给 collator，减少无效计算。"""
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # 因果语言建模中，label 由 input_ids 复制得到，不使用 MLM 遮盖。
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
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
