"""GRPO reward functions backed by the shared train/eval verifier."""

from src.math_verifier import (
    answers_equivalent,
    completion_to_text,
    extract_final_answer,
    has_required_format,
)


def correctness_reward(completions, answer=None, **kwargs):
    """按最终答案是否与标准答案完全一致返回 0/1 奖励。"""
    rewards = []
    answers = answer if answer is not None else kwargs.get("answers")

    for completion, gold in zip(completions, answers):
        rewards.append(1.0 if answers_equivalent(completion, gold) else 0.0)

    return rewards


def format_reward(completions, **kwargs):
    """鼓励模型显式使用 ``Final Answer:`` 标记最终答案。"""
    rewards = []
    for completion in completions:
        rewards.append(0.1 if has_required_format(completion) else -0.2)
    return rewards


def length_reward(completions, max_words=180, hard_max_words=220, **kwargs):
    """Apply a smooth penalty near the length limit instead of a hard cliff."""
    rewards = []
    for completion in completions:
        text = completion_to_text(completion)
        word_count = len(text.split())
        if word_count <= max_words:
            rewards.append(0.0)
        else:
            width = max(1, hard_max_words - max_words)
            severity = min(1.0, (word_count - max_words) / width)
            rewards.append(-0.1 * severity)
    return rewards


def combined_reward(completions, answer=None, **kwargs):
    """将正确性、输出格式和长度三项奖励逐样本相加。"""
    c = correctness_reward(completions, answer=answer, **kwargs)
    f = format_reward(completions, **kwargs)
    l = length_reward(completions, **kwargs)
    return [c_i + f_i + l_i for c_i, f_i, l_i in zip(c, f, l)]
