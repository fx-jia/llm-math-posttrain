import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "processed" / "dpo_pairs_sft.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rough_len(text: str) -> int:
    return len(text.split())


def main() -> None:
    rows = load_jsonl(PATH)
    chosen_lens = [rough_len(row["chosen"]) for row in rows]
    rejected_lens = [rough_len(row["rejected"]) for row in rows]
    format_rate = sum("Final Answer:" in row["rejected"] for row in rows) / len(rows)

    print(f"pairs: {len(rows)}")
    print(f"chosen length mean/median/max: {mean(chosen_lens):.1f} / {median(chosen_lens):.1f} / {max(chosen_lens)}")
    print(f"rejected length mean/median/max: {mean(rejected_lens):.1f} / {median(rejected_lens):.1f} / {max(rejected_lens)}")
    print(f"rejected format rate: {format_rate:.4f}")

    print("=" * 80)
    print("sample pair:")
    sample = rows[0]
    print("id:", sample["id"])
    print("gold:", sample["gold"])
    print("rejected_prediction:", sample["rejected_prediction"])
    print("prompt preview:")
    print(sample["prompt"][:500])
    print("chosen:")
    print(sample["chosen"][:800])
    print("rejected:")
    print(sample["rejected"][:800])


if __name__ == "__main__":
    main()
