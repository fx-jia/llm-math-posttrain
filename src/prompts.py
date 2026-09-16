"""Canonical prompt and completion templates for all training stages."""

MATH_INSTRUCTION = (
    "You are a helpful math reasoning assistant.\n"
    "Solve the following problem step by step, and put the final answer after "
    "'Final Answer:'.\n\n"
)


def build_math_prompt(question: str) -> str:
    return f"{MATH_INSTRUCTION}Problem:\n{question.strip()}\n\nSolution:\n"


def build_math_completion(reasoning: str, answer: str) -> str:
    return f"{reasoning.strip()}\nFinal Answer: {answer.strip()}"


def render_prompt_for_model(tokenizer, prompt: str, use_chat_template: bool = True) -> str:
    """Render one canonical user prompt with the model's own chat template.

    Base checkpoints without a chat template keep the raw prompt.  This lets
    the same data path support both Qwen3.5-4B-Base and instruct checkpoints.
    """
    if not use_chat_template or not getattr(tokenizer, "chat_template", None):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_sft_text(question: str, reasoning: str, answer: str) -> str:
    return build_math_prompt(question) + build_math_completion(reasoning, answer)
