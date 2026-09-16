import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rewards import (
    extract_final_answer,
    correctness_reward,
    format_reward,
    length_reward,
    combined_reward,
)


completions = [
    "We compute 48 / 2 = 24, then 48 + 24 = 72.\nFinal Answer: 72",
    "We compute something incorrectly.\nFinal Answer: 70",
    "The answer is 72",
    " ".join(["word"] * 220) + "\nFinal Answer: 72",
]
answers = ["72", "72", "72", "72"]

print("Extracted answers:")
for text in completions:
    print(extract_final_answer(text))

print("correctness_reward:", correctness_reward(completions, answer=answers))
print("format_reward:", format_reward(completions))
print("length_reward:", length_reward(completions))
print("combined_reward:", combined_reward(completions, answer=answers))

# Train-time reward and eval-time verifier share the same decimal semantics.
assert correctness_reward(["Final Answer: 26.00"], answer=["26"]) == [1.0]
