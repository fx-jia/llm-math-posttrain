import json
import sys
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.math_verifier import has_required_format

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
    format_rate = sum(has_required_format(row["rejected"]) for row in rows) / len(rows)
    sources = sorted({row.get("source", "unknown") for row in rows})

    print(f"pairs: {len(rows)}")
    print(f"chosen length mean/median/max: {mean(chosen_lens):.1f} / {median(chosen_lens):.1f} / {max(chosen_lens)}")
    print(f"rejected length mean/median/max: {mean(rejected_lens):.1f} / {median(rejected_lens):.1f} / {max(rejected_lens)}")
    print(f"rejected format rate: {format_rate:.4f}")
    print(f"pair sources: {sources}")
    if all("length_gap" in row for row in rows):
        print(f"token length gap mean/median: {mean(row['length_gap'] for row in rows):.1f} / "
              f"{median(row['length_gap'] for row in rows):.1f}")
    if all("sft_pass_rate" in row for row in rows):
        print(f"source policy pass rate mean: {mean(row['sft_pass_rate'] for row in rows):.4f}")

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
