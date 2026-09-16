"""Shared parsing and verification for numeric math answers.

The same verifier is used by data construction, preference training, RL rewards,
and evaluation.  Keeping this logic in one place prevents train/eval reward
drift, which is particularly easy to miss in RLVR experiments.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


_FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*[:=]\s*([^\n]+)", re.IGNORECASE)
_LATEX_FRAC_RE = re.compile(
    r"\\(?:d?frac)\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}"
    r"\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}"
)
_NUMBER_RE = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)
_FRACTION_RE = re.compile(
    rf"(?P<numerator>{_NUMBER_RE.pattern})\s*/\s*(?P<denominator>{_NUMBER_RE.pattern})"
)


def completion_to_text(completion) -> str:
    """Normalize the completion structures returned by different TRL versions."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])
    return str(completion)


def _boxed_contents(text: str) -> list[str]:
    """Extract balanced ``\\boxed{...}`` contents, including nested fractions."""
    results = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        index = start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            results.append(text[start : index - 1].strip())
    return results


def _normalize_latex_fraction(text: str) -> str:
    return _LATEX_FRAC_RE.sub(r"\1/\2", text)


def extract_final_answer(output: str) -> str:
    """Extract the final answer candidate without yet deciding equivalence.

    Explicit ``Final Answer:`` fields take priority, followed by the last
    ``\\boxed{...}``, and finally the last numeric expression in the output.
    """
    text = completion_to_text(output).strip()
    explicit = _FINAL_ANSWER_RE.findall(text)
    if explicit:
        candidate = explicit[-1].strip()
        boxed = _boxed_contents(candidate)
        return boxed[-1] if boxed else candidate

    boxed = _boxed_contents(text)
    if boxed:
        return boxed[-1]

    normalized = _normalize_latex_fraction(text)
    fractions = list(_FRACTION_RE.finditer(normalized))
    numbers = list(_NUMBER_RE.finditer(normalized))
    def expression(match, priority: int) -> tuple[int, int, str]:
        suffix = normalized[match.end() :]
        percent = "%" if re.match(r"\s*(?:\\%)?%?", suffix).group(0).strip() else ""
        return match.end(), priority, match.group(0) + percent

    choices = [expression(match, 1) for match in fractions]
    choices.extend(expression(match, 0) for match in numbers)
    return max(choices, default=(-1, -1, ""), key=lambda item: (item[0], item[1]))[2]


def _decimal_fraction(text: str) -> Fraction:
    cleaned = text.replace(",", "").strip()
    return Fraction(Decimal(cleaned))


def parse_numeric_answer(text: str) -> Fraction | None:
    """Parse a numeric answer into an exact rational value when possible."""
    candidate = extract_final_answer(text)
    candidate = _normalize_latex_fraction(candidate)
    candidate = candidate.replace("$", "").replace("¥", "").strip()

    is_percent = "%" in candidate or r"\%" in candidate
    candidate = candidate.replace(r"\%", "").replace("%", "")

    fraction_matches = list(_FRACTION_RE.finditer(candidate))
    try:
        if fraction_matches:
            match = fraction_matches[-1]
            denominator = _decimal_fraction(match.group("denominator"))
            if denominator == 0:
                return None
            value = _decimal_fraction(match.group("numerator")) / denominator
        else:
            numbers = list(_NUMBER_RE.finditer(candidate))
            if not numbers:
                return None
            value = _decimal_fraction(numbers[-1].group(0))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None

    return value / 100 if is_percent else value


def _fraction_to_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)

    denominator = value.denominator
    while denominator % 2 == 0:
        denominator //= 2
    while denominator % 5 == 0:
        denominator //= 5
    if denominator != 1:
        return f"{value.numerator}/{value.denominator}"

    decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
    return format(decimal_value.normalize(), "f")


def normalize_answer(text: str) -> str:
    """Return a stable representation used by logs and legacy callers."""
    numeric = parse_numeric_answer(text)
    if numeric is not None:
        return _fraction_to_text(numeric)

    candidate = extract_final_answer(text)
    return candidate.strip().rstrip(".").casefold()


def answers_equivalent(prediction: str, reference: str) -> bool:
    """Compare answers by exact numeric value, falling back to normalized text."""
    pred_value = parse_numeric_answer(prediction)
    ref_value = parse_numeric_answer(reference)
    if pred_value is not None and ref_value is not None:
        return pred_value == ref_value
    return normalize_answer(prediction) == normalize_answer(reference)


def has_required_format(text: str) -> bool:
    """Whether a completion has an explicit machine-readable final answer."""
    output = completion_to_text(text)
    return bool(_FINAL_ANSWER_RE.search(output) or _boxed_contents(output))
