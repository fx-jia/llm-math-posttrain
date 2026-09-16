import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prompts import build_math_prompt

FULL_TRAIN_PATH = ROOT / "data" / "processed" / "gsm8k_train.jsonl"
CORE_TRAIN_PATH = ROOT / "data" / "processed" / "gsm8k_train_core.jsonl"
TRAIN_PATH = CORE_TRAIN_PATH if CORE_TRAIN_PATH.exists() else FULL_TRAIN_PATH
OUTPUT_PATH = ROOT / "data" / "processed" / "grpo_train.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    with TRAIN_PATH.open("r", encoding="utf-8") as fin:
        rows = [json.loads(line) for line in fin]
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for row in rows:
            record = {
                "id": row["id"],
                "prompt": build_math_prompt(row["question"]),
                "answer": row["answer"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"saved examples: {len(rows)}")
    print(f"shuffle={args.shuffle}, seed={args.seed}")
    print(f"output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
