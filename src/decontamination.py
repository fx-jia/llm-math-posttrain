"""Lightweight lexical train/evaluation overlap detection."""

from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"\\[a-zA-Z]+|[a-zA-Z]+|\d+(?:\.\d+)?|[^\s\w]", re.UNICODE)


def normalized_tokens(text: str) -> tuple[str, ...]:
    """Tokenize prose and LaTeX deterministically for overlap checks."""
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text)))


def normalized_text(text: str) -> str:
    return " ".join(normalized_tokens(text))


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(normalized_tokens(left))
    right_tokens = set(normalized_tokens(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def find_overlaps(
    train_rows: list[dict],
    eval_rows: list[dict],
    threshold: float = 0.8,
) -> list[dict]:
    """Return the strongest suspicious evaluation match for each train row."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    eval_items = [
        (
            row.get("id", f"eval-{index}"),
            str(row.get("question", row.get("problem", ""))),
            row.get("benchmark", row.get("source", "unknown")),
        )
        for index, row in enumerate(eval_rows)
    ]
    overlaps = []
    for index, row in enumerate(train_rows):
        question = str(row.get("question", row.get("problem", "")))
        train_normalized = normalized_text(question)
        best = None
        for eval_id, eval_question, benchmark in eval_items:
            exact = bool(train_normalized) and train_normalized == normalized_text(eval_question)
            similarity = 1.0 if exact else jaccard_similarity(question, eval_question)
            if best is None or similarity > best["similarity"]:
                best = {
                    "eval_id": eval_id,
                    "benchmark": benchmark,
                    "similarity": similarity,
                    "exact_normalized_match": exact,
                }
        if best and best["similarity"] >= threshold:
            overlaps.append(
                {
                    "train_index": index,
                    "train_id": row.get("id", f"train-{index}"),
                    **best,
                }
            )
    return sorted(overlaps, key=lambda item: (-item["similarity"], str(item["train_id"])))
