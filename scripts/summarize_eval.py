import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def summarize(path: Path) -> dict:
    rows = load_jsonl(path)
    n = len(rows)
    return {
        "file": str(path),
        "examples": n,
        "exact_match": sum(row["correct"] for row in rows) / n,
        "format_rate": sum(row["format_ok"] for row in rows) / n,
        "avg_latency_sec": sum(row["latency"] for row in rows) / n,
        "avg_output_tokens": sum(row["output_tokens"] for row in rows) / n,
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/summarize_eval.py <eval_jsonl> ...")

    summaries = [summarize(Path(arg)) for arg in sys.argv[1:]]

    print("| file | examples | exact_match | format_rate | avg_latency_sec | avg_output_tokens |")
    print("|---|---:|---:|---:|---:|---:|")
    for item in summaries:
        print(
            f"| {item['file']} | {item['examples']} | "
            f"{item['exact_match']:.4f} | {item['format_rate']:.4f} | "
            f"{item['avg_latency_sec']:.2f} | {item['avg_output_tokens']:.1f} |"
        )


if __name__ == "__main__":
    main()
