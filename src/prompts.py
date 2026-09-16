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


def build_sft_text(question: str, reasoning: str, answer: str) -> str:
    return build_math_prompt(question) + build_math_completion(reasoning, answer)
