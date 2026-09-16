"""Dependency-free construction of response-only SFT token labels."""

from src.prompts import build_math_completion, build_math_prompt, render_prompt_for_model


def encode_response_only(
    row: dict,
    tokenizer,
    max_length: int,
    use_chat_template: bool = False,
) -> dict | None:
    """Tokenize prompt/completion separately and mask every prompt token."""
    if "prompt" in row and "completion" in row:
        prompt = row["prompt"]
        completion = row["completion"]
    else:
        prompt = build_math_prompt(row["question"])
        completion = build_math_completion(row["reasoning"], row["answer"])

    prompt = render_prompt_for_model(tokenizer, prompt, use_chat_template)
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False, truncation=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [tokenizer.eos_token_id]

    if len(prompt_ids) + len(completion_ids) > max_length:
        return None
    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }
