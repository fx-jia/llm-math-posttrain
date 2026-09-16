import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.statistics import wilson_interval


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def summarize(path: Path) -> dict:
    rows = load_jsonl(path)
    n = len(rows)
    correct = sum(row["correct"] for row in rows)
    ci_low, ci_high = wilson_interval(correct, n)
    total_tokens = sum(row["output_tokens"] for row in rows)
    return {
        "file": str(path),
        "examples": n,
        "exact_match": correct / n,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "format_rate": sum(row["format_ok"] for row in rows) / n,
        "avg_latency_sec": sum(row["latency"] for row in rows) / n,
        "avg_output_tokens": sum(row["output_tokens"] for row in rows) / n,
        "tokens_per_correct": total_tokens / correct if correct else float("inf"),
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/summarize_eval.py <eval_jsonl> ...")

    summaries = [summarize(Path(arg)) for arg in sys.argv[1:]]

    print("| file | examples | exact_match (95% CI) | format_rate | avg_latency_sec | avg_tokens | tokens/correct |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for item in summaries:
        print(
            f"| {item['file']} | {item['examples']} | "
            f"{item['exact_match']:.4f} [{item['ci_low']:.4f}, {item['ci_high']:.4f}] | "
            f"{item['format_rate']:.4f} | {item['avg_latency_sec']:.2f} | "
            f"{item['avg_output_tokens']:.1f} | {item['tokens_per_correct']:.1f} |"
        )


if __name__ == "__main__":
    main()
