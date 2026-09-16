"""Prepare deterministic OpenR1 SFT and DAPO-Math RLVR datasets for V3."""

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.math_verifier import has_required_format
from src.prompts import build_math_prompt
from src.run_manifest import write_manifest


OPENR1_DATASET = "open-r1/OpenR1-Math-220k"
DAPO_DATASET = "open-r1/DAPO-Math-17k-Processed"
SFT_OUTPUT = ROOT / "data" / "processed" / "openr1_sft_v3.jsonl"
RLVR_OUTPUT = ROOT / "data" / "processed" / "dapo_math_v3.jsonl"


def _message_text(messages, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != role:
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [item.get("text", "") for item in content if isinstance(item, dict)]
            return "\n".join(part for part in parts if part).strip()
    return ""


def normalize_openr1_row(row: dict, index: int, min_correctness: int = 2) -> dict | None:
    """Convert one OpenR1 row without importing datasets or model libraries."""
    correctness = row.get("correctness_count")
    if correctness is not None and int(correctness) < min_correctness:
        return None

    question = str(row.get("problem") or _message_text(row.get("messages"), "user")).strip()
    answer = str(row.get("answer") or "").strip()
    completion = str(
        _message_text(row.get("messages"), "assistant") or row.get("solution") or ""
    ).strip()
    if not question or not answer or not completion:
        return None
    if not has_required_format(completion):
        completion = f"{completion}\nFinal Answer: {answer}"

    return {
        "id": str(row.get("uuid") or f"openr1-{index}"),
        "source": str(row.get("source") or OPENR1_DATASET),
        "category": str(row.get("problem_type") or row.get("question_type") or "unknown"),
        "question": question,
        "prompt": build_math_prompt(question),
        "completion": completion,
        "answer": answer,
        "correctness_count": correctness,
    }


def normalize_dapo_row(row: dict, index: int) -> dict | None:
    question = str(row.get("prompt") or row.get("problem") or "").strip()
    answer = str(row.get("solution") or row.get("answer") or "").strip()
    if not question or not answer:
        return None
    return {
        "id": str(row.get("id") or f"dapo-{index}"),
        "source": str(row.get("source") or DAPO_DATASET),
        "question": question,
        "prompt": build_math_prompt(question),
        "answer": answer,
    }


def _prepare(dataset, normalizer, seed: int, limit: int | None, **kwargs) -> list[dict]:
    rows = []
    for index, source_row in enumerate(dataset):
        row = normalizer(dict(source_row), index, **kwargs)
        if row is not None:
            rows.append(row)
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit is not None else rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openr1-revision", default="main")
    parser.add_argument("--dapo-revision", default="main")
    parser.add_argument("--sft-limit", type=int, default=40000)
    parser.add_argument("--rlvr-limit", type=int, default=None)
    parser.add_argument("--min-correctness", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-output", default=str(SFT_OUTPUT.relative_to(ROOT)))
    parser.add_argument("--rlvr-output", default=str(RLVR_OUTPUT.relative_to(ROOT)))
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {OPENR1_DATASET}@{args.openr1_revision}")
    openr1 = load_dataset(OPENR1_DATASET, split="train", revision=args.openr1_revision)
    print(f"Loading {DAPO_DATASET}@{args.dapo_revision} config=all")
    dapo = load_dataset(
        DAPO_DATASET,
        "all",
        split="train",
        revision=args.dapo_revision,
    )

    sft_rows = _prepare(
        openr1,
        normalize_openr1_row,
        args.seed,
        args.sft_limit,
        min_correctness=args.min_correctness,
    )
    rlvr_rows = _prepare(dapo, normalize_dapo_row, args.seed, args.rlvr_limit)
    sft_output = ROOT / args.sft_output if not Path(args.sft_output).is_absolute() else Path(args.sft_output)
    rlvr_output = ROOT / args.rlvr_output if not Path(args.rlvr_output).is_absolute() else Path(args.rlvr_output)
    _write_jsonl(sft_output, sft_rows)
    _write_jsonl(rlvr_output, rlvr_rows)

    write_manifest(
        sft_output.with_suffix(".manifest.json"),
        ROOT,
        vars(args),
        dataset_name=OPENR1_DATASET,
        dataset_revision=args.openr1_revision,
        dataset_fingerprint=getattr(openr1, "_fingerprint", None),
        examples=len(sft_rows),
    )
    write_manifest(
        rlvr_output.with_suffix(".manifest.json"),
        ROOT,
        vars(args),
        dataset_name=DAPO_DATASET,
        dataset_config="all",
        dataset_revision=args.dapo_revision,
        dataset_fingerprint=getattr(dapo, "_fingerprint", None),
        examples=len(rlvr_rows),
    )
    print(f"SFT: {len(sft_rows)} -> {sft_output}")
    print(f"RLVR: {len(rlvr_rows)} -> {rlvr_output}")


if __name__ == "__main__":
    main()
