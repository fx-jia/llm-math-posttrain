import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "gsm8k_train.jsonl"
OUTPUT_PATH = ROOT / "data" / "processed" / "grpo_train.jsonl"


def build_prompt(question: str) -> str:
    return (
        "You are a helpful math reasoning assistant.\n"
        "Solve the following problem step by step, and put the final answer after 'Final Answer:'.\n\n"
        f"Problem:\n{question.strip()}\n\n"
        "Solution:\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=512)
    args = parser.parse_args()

    count = 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with TRAIN_PATH.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            record = {
                "id": row["id"],
                "prompt": build_prompt(row["question"]),
                "answer": row["answer"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count >= args.limit:
                break

    print(f"saved examples: {count}")
    print(f"output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
