"""Download and normalize the frozen V3 evaluation suite."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.run_manifest import write_manifest


BENCHMARKS = {
    "hmmt_feb_2026": {"dataset": "MathArena/hmmt_feb_2026", "split": "train"},
    "aime_2026": {"dataset": "MathArena/aime_2026", "split": "train"},
    "apex_2025": {"dataset": "MathArena/apex_2025", "split": "train"},
    "math_500": {"dataset": "HuggingFaceH4/MATH-500", "split": "test"},
}
MISSING_MARKERS = {"", "n/a", "na", "none", "null", "-"}


def normalize_benchmark_row(row: dict, index: int, benchmark: str) -> dict | None:
    question = str(row.get("problem") or row.get("question") or "").strip()
    answer = str(row.get("answer") or row.get("solution") or "").strip()
    if question.casefold() in MISSING_MARKERS or answer.casefold() in MISSING_MARKERS:
        return None
    identifier = row.get("problem_idx", row.get("unique_id", row.get("id", index)))
    return {
        "id": f"{benchmark}-{identifier}",
        "benchmark": benchmark,
        "source": BENCHMARKS[benchmark]["dataset"],
        "category": str(row.get("problem_type") or row.get("subject") or "unknown"),
        "question": question,
        "answer": answer,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", choices=sorted(BENCHMARKS), default=list(BENCHMARKS))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    from datasets import load_dataset

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    combined_matharena = []
    component_fingerprints = {}
    for benchmark in args.benchmarks:
        spec = BENCHMARKS[benchmark]
        print(f"Loading {spec['dataset']}@{args.revision} split={spec['split']}")
        dataset = load_dataset(spec["dataset"], split=spec["split"], revision=args.revision)
        component_fingerprints[benchmark] = getattr(dataset, "_fingerprint", None)
        rows = [
            normalized
            for index, row in enumerate(dataset)
            if (normalized := normalize_benchmark_row(dict(row), index, benchmark)) is not None
        ]
        path = output_dir / f"{benchmark}.jsonl"
        _write_jsonl(path, rows)
        write_manifest(
            path.with_suffix(".manifest.json"),
            ROOT,
            vars(args),
            benchmark=benchmark,
            dataset_name=spec["dataset"],
            dataset_revision=args.revision,
            dataset_fingerprint=component_fingerprints[benchmark],
            examples=len(rows),
        )
        if benchmark != "math_500":
            combined_matharena.extend(rows)
        print(f"{benchmark}: {len(rows)} -> {path}")

    if combined_matharena:
        combined_path = output_dir / "matharena_v3.jsonl"
        _write_jsonl(combined_path, combined_matharena)
        write_manifest(
            combined_path.with_suffix(".manifest.json"),
            ROOT,
            vars(args),
            component_fingerprints=component_fingerprints,
            examples=len(combined_matharena),
        )
        print(f"MathArena combined: {len(combined_matharena)} -> {combined_path}")


if __name__ == "__main__":
    main()
