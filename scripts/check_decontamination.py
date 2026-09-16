"""Audit and optionally remove lexical overlap with the V3 evaluation suite."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.decontamination import find_overlaps
from src.run_manifest import write_manifest


DEFAULT_EVAL_PATHS = [
    "data/processed/hmmt_feb_2026.jsonl",
    "data/processed/aime_2026.jsonl",
    "data/processed/apex_2025.jsonl",
    "data/processed/math_500.jsonl",
]


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/processed/openr1_sft_v3.jsonl")
    parser.add_argument("--eval-paths", nargs="+", default=DEFAULT_EVAL_PATHS)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--report", default="data/processed/decontamination_v3.json")
    parser.add_argument("--filtered-output")
    args = parser.parse_args()

    train_path = _resolve(args.train_path)
    eval_paths = [_resolve(value) for value in args.eval_paths]
    train_rows = _load_jsonl(train_path)
    eval_rows = [row for path in eval_paths for row in _load_jsonl(path)]
    overlaps = find_overlaps(train_rows, eval_rows, args.threshold)
    contaminated_indices = {item["train_index"] for item in overlaps}
    report = {
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "threshold": args.threshold,
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
    }
    report_path = _resolve(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.filtered_output:
        filtered_path = _resolve(args.filtered_output)
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        with filtered_path.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(train_rows):
                if index not in contaminated_indices:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Filtered train data: {len(train_rows) - len(overlaps)} -> {filtered_path}")

    write_manifest(
        report_path.with_suffix(".manifest.json"),
        ROOT,
        vars(args),
        train_path=str(train_path),
        overlap_count=len(overlaps),
    )
    print(f"Potential overlaps: {len(overlaps)} -> {report_path}")


if __name__ == "__main__":
    main()
