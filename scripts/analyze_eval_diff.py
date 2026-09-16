import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.statistics import exact_mcnemar_p


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def short(text: str, n: int = 500) -> str:
    text = text.replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args()

    a = load_jsonl(Path(args.file_a))
    b = load_jsonl(Path(args.file_b))

    common_ids = sorted(set(a) & set(b))
    if not common_ids:
        raise SystemExit("No common ids found.")

    both_correct = []
    a_correct_b_wrong = []
    a_wrong_b_correct = []
    both_wrong = []

    for item_id in common_ids:
        ra = a[item_id]
        rb = b[item_id]

        if ra["correct"] and rb["correct"]:
            both_correct.append(item_id)
        elif ra["correct"] and not rb["correct"]:
            a_correct_b_wrong.append(item_id)
        elif not ra["correct"] and rb["correct"]:
            a_wrong_b_correct.append(item_id)
        else:
            both_wrong.append(item_id)

    n = len(common_ids)
    print(f"Compared files:")
    print(f"{args.name_a}: {args.file_a}")
    print(f"{args.name_b}: {args.file_b}")
    print("=" * 80)
    print(f"common examples: {n}")
    print(f"both_correct: {len(both_correct)}")
    print(f"{args.name_a}_correct_{args.name_b}_wrong: {len(a_correct_b_wrong)}")
    print(f"{args.name_a}_wrong_{args.name_b}_correct: {len(a_wrong_b_correct)}")
    print(f"both_wrong: {len(both_wrong)}")
    print(
        "exact_mcnemar_p: "
        f"{exact_mcnemar_p(len(a_correct_b_wrong), len(a_wrong_b_correct)):.6f}"
    )

    avg_tokens_a = sum(a[i]["output_tokens"] for i in common_ids) / n
    avg_tokens_b = sum(b[i]["output_tokens"] for i in common_ids) / n
    format_a = sum(a[i]["format_ok"] for i in common_ids) / n
    format_b = sum(b[i]["format_ok"] for i in common_ids) / n

    print("=" * 80)
    print(f"{args.name_a} avg_output_tokens: {avg_tokens_a:.1f}")
    print(f"{args.name_b} avg_output_tokens: {avg_tokens_b:.1f}")
    print(f"{args.name_a} format_rate: {format_a:.4f}")
    print(f"{args.name_b} format_rate: {format_b:.4f}")

    print("=" * 80)
    print(f"Examples where {args.name_a} was correct but {args.name_b} became wrong:")
    for item_id in a_correct_b_wrong[: args.show]:
        ra = a[item_id]
        rb = b[item_id]
        print("-" * 80)
        print(f"id: {item_id}")
        print(f"gold: {ra['gold']}")
        print(f"{args.name_a} pred: {ra['prediction']} | format={ra['format_ok']} | tokens={ra['output_tokens']}")
        print(f"{args.name_b} pred: {rb['prediction']} | format={rb['format_ok']} | tokens={rb['output_tokens']}")
        print("question:", short(ra["question"], 350))
        print(f"{args.name_a} output:", short(ra["output"], 500))
        print(f"{args.name_b} output:", short(rb["output"], 500))

    print("=" * 80)
    print(f"Examples where {args.name_a} was wrong but {args.name_b} became correct:")
    for item_id in a_wrong_b_correct[: args.show]:
        ra = a[item_id]
        rb = b[item_id]
        print("-" * 80)
        print(f"id: {item_id}")
        print(f"gold: {ra['gold']}")
        print(f"{args.name_a} pred: {ra['prediction']} | format={ra['format_ok']} | tokens={ra['output_tokens']}")
        print(f"{args.name_b} pred: {rb['prediction']} | format={rb['format_ok']} | tokens={rb['output_tokens']}")
        print("question:", short(ra["question"], 350))
        print(f"{args.name_a} output:", short(ra["output"], 500))
        print(f"{args.name_b} output:", short(rb["output"], 500))


if __name__ == "__main__":
    main()
