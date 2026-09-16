import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prompts import build_sft_text

PROCESSED_DIR = ROOT / "data" / "processed"


def extract_final_answer(raw_answer: str) -> str:
    """Extract the final numeric answer after the GSM8K delimiter '####'."""
    if "####" not in raw_answer:
        return ""
    final = raw_answer.split("####")[-1].strip()
    final = final.replace(",", "")
    return final


def extract_reasoning(raw_answer: str) -> str:
    """Extract chain-of-thought reasoning before the final GSM8K answer."""
    if "####" not in raw_answer:
        return raw_answer.strip()
    return raw_answer.split("####")[0].strip()


def convert_split(split_name: str, records) -> list[dict]:
    converted = []
    for idx, item in enumerate(records):
        question = item["question"].strip()
        raw_answer = item["answer"].strip()
        reasoning = extract_reasoning(raw_answer)
        answer = extract_final_answer(raw_answer)

        if not question or not reasoning or not answer:
            continue

        converted.append(
            {
                "id": f"gsm8k_{split_name}_{idx}",
                "source": "gsm8k",
                "question": question,
                "reasoning": reasoning,
                "answer": answer,
                "sft_text": build_sft_text(question, reasoning, answer),
            }
        )
    return converted


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def check_outputs() -> None:
    train_path = PROCESSED_DIR / "gsm8k_train.jsonl"
    train_core_path = PROCESSED_DIR / "gsm8k_train_core.jsonl"
    dev_path = PROCESSED_DIR / "gsm8k_dev.jsonl"
    test_path = PROCESSED_DIR / "gsm8k_test.jsonl"

    for path in (train_path, train_core_path, dev_path, test_path):
        assert path.exists(), f"Missing {path}"

    train_rows = load_jsonl(train_path)
    train_core_rows = load_jsonl(train_core_path)
    dev_rows = load_jsonl(dev_path)
    test_rows = load_jsonl(test_path)

    required_keys = {"id", "source", "question", "reasoning", "answer", "sft_text"}
    for name, rows in [
        ("train", train_rows),
        ("train_core", train_core_rows),
        ("dev", dev_rows),
        ("test", test_rows),
    ]:
        assert rows, f"{name} split is empty"
        for row in rows[:20]:
            assert required_keys.issubset(row.keys()), row.keys()
            assert "Final Answer:" in row["sft_text"]
            assert row["answer"].strip() != ""

    print(f"Check passed.")
    print(f"train examples: {len(train_rows)}")
    print(f"train core examples: {len(train_core_rows)}")
    print(f"dev examples: {len(dev_rows)}")
    print(f"test examples: {len(test_rows)}")
    print("sample question:", train_rows[0]["question"][:120].replace("\n", " "))
    print("sample answer:", train_rows[0]["answer"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--from-existing",
        action="store_true",
        help="Create the V2 train/dev split from existing processed JSONL without downloading.",
    )
    parser.add_argument("--dev-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return

    if args.from_existing:
        train_rows = load_jsonl(PROCESSED_DIR / "gsm8k_train.jsonl")
        test_rows = load_jsonl(PROCESSED_DIR / "gsm8k_test.jsonl")
    else:
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main")
        train_rows = convert_split("train", dataset["train"])
        test_rows = convert_split("test", dataset["test"])

    if not 0 < args.dev_size < len(train_rows):
        raise ValueError("--dev-size must be between 1 and the number of training examples - 1")
    shuffled_indices = list(range(len(train_rows)))
    random.Random(args.seed).shuffle(shuffled_indices)
    dev_indices = set(shuffled_indices[: args.dev_size])
    dev_rows = [row for index, row in enumerate(train_rows) if index in dev_indices]
    train_core_rows = [row for index, row in enumerate(train_rows) if index not in dev_indices]

    save_jsonl(PROCESSED_DIR / "gsm8k_train.jsonl", train_rows)
    save_jsonl(PROCESSED_DIR / "gsm8k_train_core.jsonl", train_core_rows)
    save_jsonl(PROCESSED_DIR / "gsm8k_dev.jsonl", dev_rows)
    save_jsonl(PROCESSED_DIR / "gsm8k_test.jsonl", test_rows)

    print(f"Saved train examples: {len(train_rows)}")
    print(f"Saved train core examples: {len(train_core_rows)}")
    print(f"Saved dev examples: {len(dev_rows)} (seed={args.seed})")
    print(f"Saved test examples: {len(test_rows)}")
    print(f"Output directory: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
