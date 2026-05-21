import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "gsm8k_train.jsonl"
OUTPUT_PATH = ROOT / "data" / "processed" / "dpo_pairs_sft.jsonl"

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
ADAPTER_DIR = os.environ.get("SFT_ADAPTER_DIR", str(ROOT / "outputs" / "sft_lora_r8_full"))


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.rstrip(".")
    return text


def extract_final_answer(output: str) -> str:
    match = re.search(r"Final Answer:\s*([^\n]+)", output)
    if match:
        candidate = match.group(1).strip()
    else:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", output.replace(",", ""))
        candidate = numbers[-1] if numbers else ""

    numbers = re.findall(r"-?\d+(?:\.\d+)?", candidate.replace(",", ""))
    if numbers:
        return normalize_answer(numbers[-1])
    return normalize_answer(candidate)


def build_prompt(question: str) -> str:
    return (
        "You are a helpful math reasoning assistant.\n"
        "Solve the following problem step by step, and put the final answer after 'Final Answer:'.\n\n"
        f"Problem:\n{question.strip()}\n\n"
        "Solution:\n"
    )


def build_chosen(row: dict) -> str:
    return f"{row['reasoning'].strip()}\nFinal Answer: {row['answer'].strip()}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-limit", type=int, default=800)
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

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

    rows = load_jsonl(TRAIN_PATH, limit=args.input_limit)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pairs = []
    start_time = time.time()

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            prompt = build_prompt(row["question"])
            gold = normalize_answer(row["answer"])

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            rejected = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            pred = extract_final_answer(rejected)

            # DPO needs a clear preference. We only keep wrong sampled answers.
            if pred and pred != gold and "Final Answer:" in rejected:
                record = {
                    "id": row["id"],
                    "source": "gsm8k_sft_sampled_wrong",
                    "prompt": prompt,
                    "chosen": build_chosen(row),
                    "rejected": rejected,
                    "gold": gold,
                    "rejected_prediction": pred,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                pairs.append(record)

            if idx % 50 == 0:
                print(f"processed={idx}, pairs={len(pairs)}")

            if len(pairs) >= args.max_pairs:
                break

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"processed examples: {idx}")
    print(f"saved pairs: {len(pairs)}")
    print(f"output path: {OUTPUT_PATH}")
    print(f"elapsed seconds: {elapsed:.2f}")
    if torch.cuda.is_available():
        print(f"peak_cuda_memory_gb: {torch.cuda.max_memory_allocated() / 1024**3:.2f}")


if __name__ == "__main__":
    main()
