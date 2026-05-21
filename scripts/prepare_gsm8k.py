import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
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


def build_sft_text(question: str, reasoning: str, answer: str) -> str:
    """Build a stable instruction format for supervised fine-tuning."""
    return (
        "You are a helpful math reasoning assistant.\n"
        "Solve the following problem step by step, and put the final answer after 'Final Answer:'.\n\n"
        f"Problem:\n{question.strip()}\n\n"
        f"Solution:\n{reasoning.strip()}\n"
        f"Final Answer: {answer.strip()}"
    )


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
    test_path = PROCESSED_DIR / "gsm8k_test.jsonl"

    assert train_path.exists(), f"Missing {train_path}"
    assert test_path.exists(), f"Missing {test_path}"

    train_rows = load_jsonl(train_path)
    test_rows = load_jsonl(test_path)

    required_keys = {"id", "source", "question", "reasoning", "answer", "sft_text"}
    for name, rows in [("train", train_rows), ("test", test_rows)]:
        assert rows, f"{name} split is empty"
        for row in rows[:20]:
            assert required_keys.issubset(row.keys()), row.keys()
            assert "Final Answer:" in row["sft_text"]
            assert row["answer"].strip() != ""

    print(f"Check passed.")
    print(f"train examples: {len(train_rows)}")
    print(f"test examples: {len(test_rows)}")
    print("sample question:", train_rows[0]["question"][:120].replace("\n", " "))
    print("sample answer:", train_rows[0]["answer"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return

    dataset = load_dataset("openai/gsm8k", "main")

    train_rows = convert_split("train", dataset["train"])
    test_rows = convert_split("test", dataset["test"])

    save_jsonl(PROCESSED_DIR / "gsm8k_train.jsonl", train_rows)
    save_jsonl(PROCESSED_DIR / "gsm8k_test.jsonl", test_rows)

    print(f"Saved train examples: {len(train_rows)}")
    print(f"Saved test examples: {len(test_rows)}")
    print(f"Output directory: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
