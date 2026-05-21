import json
import re
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "gsm8k_train.jsonl"
TEST_PATH = ROOT / "data" / "processed" / "gsm8k_test.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rough_token_count(text: str) -> int:
    # This is not tokenizer-accurate. It is a cheap approximation for early data inspection.
    return len(text.split())


def is_simple_numeric_answer(answer: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", answer.strip()))


def describe_split(name: str, rows: list[dict]) -> None:
    sft_lengths = [rough_token_count(row["sft_text"]) for row in rows]
    question_lengths = [rough_token_count(row["question"]) for row in rows]
    reasoning_lengths = [rough_token_count(row["reasoning"]) for row in rows]
    numeric_answers = [is_simple_numeric_answer(row["answer"]) for row in rows]
    format_ok = ["Final Answer:" in row["sft_text"] for row in rows]

    print(f"== {name} ==")
    print(f"examples: {len(rows)}")
    print(f"sft_text length mean/median/max: {mean(sft_lengths):.1f} / {median(sft_lengths):.1f} / {max(sft_lengths)}")
    print(f"question length mean/median/max: {mean(question_lengths):.1f} / {median(question_lengths):.1f} / {max(question_lengths)}")
    print(f"reasoning length mean/median/max: {mean(reasoning_lengths):.1f} / {median(reasoning_lengths):.1f} / {max(reasoning_lengths)}")
    print(f"simple numeric answer rate: {sum(numeric_answers) / len(rows):.4f}")
    print(f"format compliance rate: {sum(format_ok) / len(rows):.4f}")

    longest_idx = max(range(len(rows)), key=lambda i: sft_lengths[i])
    longest = rows[longest_idx]
    print(f"longest sample id: {longest['id']}")
    print(f"longest sample rough length: {sft_lengths[longest_idx]}")
    print()


def show_samples(rows: list[dict], n: int = 2) -> None:
    print("== Sample records ==")
    for row in rows[:n]:
        print(f"id: {row['id']}")
        print(f"question: {row['question']}")
        print(f"answer: {row['answer']}")
        print("sft_text preview:")
        print(row["sft_text"][:800])
        print("-" * 80)


def main() -> None:
    train_rows = load_jsonl(TRAIN_PATH)
    test_rows = load_jsonl(TEST_PATH)

    describe_split("train", train_rows)
    describe_split("test", test_rows)
    show_samples(train_rows, n=2)


if __name__ == "__main__":
    main()
