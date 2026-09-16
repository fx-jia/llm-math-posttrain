"""Small dependency-free statistical helpers for paired model evaluation."""

from __future__ import annotations

import math


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def exact_mcnemar_p(a_correct_b_wrong: int, a_wrong_b_correct: int) -> float:
    """Two-sided exact McNemar p-value using the discordant pairs."""
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if discordant == 0:
        return 1.0
    tail = min(a_correct_b_wrong, a_wrong_b_correct)
    one_sided = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * one_sided)


def estimate_pass_at_k(samples: int, correct: int, k: int) -> float:
    """Unbiased pass@k estimate from ``samples`` independent candidates."""
    if not 1 <= k <= samples:
        raise ValueError("k must satisfy 1 <= k <= samples")
    if correct <= 0:
        return 0.0
    if samples - correct < k:
        return 1.0
    failure_probability = 1.0
    for index in range(k):
        failure_probability *= (samples - correct - index) / (samples - index)
    return 1.0 - failure_probability
