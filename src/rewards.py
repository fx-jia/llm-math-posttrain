"""GRPO 训练使用的输出解析与奖励函数。"""

import re


def completion_to_text(completion) -> str:
    """将 TRL 不同版本可能返回的 completion 结构统一为文本。"""
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


def normalize_answer(text: str) -> str:
    """移除不影响数值语义的常见格式字符。"""
    text = str(text).strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.rstrip(".")
    return text


def extract_final_answer(output: str) -> str:
    """优先解析 ``Final Answer:`` 字段，否则回退到最后一个数字。"""
    output = completion_to_text(output)
    match = re.search(r"Final Answer:\s*([^\n]+)", output)
    if match:
        candidate = match.group(1).strip()
    else:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", output.replace(",", ""))
        candidate = numbers[-1] if numbers else ""

    numbers = re.findall(r"-?\d+(?:\.\d+)?", candidate.replace(",", ""))
    if numbers:
        return normalize_answer(numbers[-1])
    return normalize_answer(candidate)


def correctness_reward(completions, answer=None, **kwargs):
    """按最终答案是否与标准答案完全一致返回 0/1 奖励。"""
    rewards = []
    answers = answer if answer is not None else kwargs.get("answers")

    for completion, gold in zip(completions, answers):
        pred = extract_final_answer(completion)
        gold = normalize_answer(gold)
        rewards.append(1.0 if pred == gold else 0.0)

    return rewards


def format_reward(completions, **kwargs):
    """鼓励模型显式使用 ``Final Answer:`` 标记最终答案。"""
    rewards = []
    for completion in completions:
        text = completion_to_text(completion)
        rewards.append(0.1 if "Final Answer:" in text else -0.2)
    return rewards


def length_reward(completions, max_words=180, **kwargs):
    """对过长回答施加轻微惩罚，抑制不必要的冗长推理。"""
    rewards = []
    for completion in completions:
        text = completion_to_text(completion)
        word_count = len(text.split())
        rewards.append(-0.1 if word_count > max_words else 0.0)
    return rewards


def combined_reward(completions, answer=None, **kwargs):
    """将正确性、输出格式和长度三项奖励逐样本相加。"""
    c = correctness_reward(completions, answer=answer, **kwargs)
    f = format_reward(completions, **kwargs)
    l = length_reward(completions, **kwargs)
    return [c_i + f_i + l_i for c_i, f_i, l_i in zip(c, f, l)]
