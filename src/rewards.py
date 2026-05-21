import re


def completion_to_text(completion) -> str:
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
    text = str(text).strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.rstrip(".")
    return text


def extract_final_answer(output: str) -> str:
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
    rewards = []
    answers = answer if answer is not None else kwargs.get("answers")

    for completion, gold in zip(completions, answers):
        pred = extract_final_answer(completion)
        gold = normalize_answer(gold)
        rewards.append(1.0 if pred == gold else 0.0)

    return rewards


def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion_to_text(completion)
        rewards.append(0.1 if "Final Answer:" in text else -0.2)
    return rewards


def length_reward(completions, max_words=180, **kwargs):
    rewards = []
    for completion in completions:
        text = completion_to_text(completion)
        word_count = len(text.split())
        rewards.append(-0.1 if word_count > max_words else 0.0)
    return rewards


def combined_reward(completions, answer=None, **kwargs):
    c = correctness_reward(completions, answer=answer, **kwargs)
    f = format_reward(completions, **kwargs)
    l = length_reward(completions, **kwargs)
    return [c_i + f_i + l_i for c_i, f_i, l_i in zip(c, f, l)]
