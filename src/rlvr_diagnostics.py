"""Parse TRL text logs and summarize RLVR signal quality."""

from __future__ import annotations

import ast
from statistics import mean


def parse_metric_records(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        start = line.find("{")
        end = line.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            record = ast.literal_eval(line[start : end + 1])
        except (SyntaxError, ValueError):
            continue
        if isinstance(record, dict) and "reward_std" in record:
            records.append(record)
    return records


def _number(record: dict, key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(records: list[dict], key: str) -> float | None:
    values = [_number(record, key) for record in records]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def summarize_rlvr(records: list[dict]) -> dict:
    if not records:
        return {"optimizer_steps": 0}
    zero_variance = sum((_number(record, "frac_reward_zero_std") or 0.0) >= 1.0 for record in records)
    summary = {
        "optimizer_steps": len(records),
        "zero_variance_steps": zero_variance,
        "zero_variance_step_rate": zero_variance / len(records),
        "mean_reward": _mean(records, "reward"),
        "mean_reward_std": _mean(records, "reward_std"),
        "mean_entropy": _mean(records, "entropy"),
        "mean_kl": _mean(records, "kl"),
        "mean_completion_length": _mean(records, "completions/mean_length"),
        "mean_clipped_completion_ratio": _mean(records, "completions/clipped_ratio"),
        "mean_clip_ratio": _mean(records, "clip_ratio/region_mean"),
    }
    warnings = []
    if summary["zero_variance_step_rate"] > 0.10:
        warnings.append("zero-variance step rate exceeds 10%; refresh or tighten frontier sampling")
    clipped = summary["mean_clipped_completion_ratio"]
    if clipped is not None and clipped > 0.05:
        warnings.append("truncated completion ratio exceeds 5%; increase budget or inspect overlong behavior")
    summary["warnings"] = warnings
    return summary
