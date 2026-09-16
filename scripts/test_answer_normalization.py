import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.math_verifier import answers_equivalent, normalize_answer


cases = [
    ("26.00", "26"),
    ("88.00", "88"),
    ("$1,430", "1430"),
    ("Final Answer: 72.", "72"),
    ("5.94", "5.94"),
    ("14.67", "14.67"),
    ("Final Answer: 3/4", "0.75"),
    (r"\boxed{\frac{1}{3}}", "1/3"),
    ("Final Answer: 1.2e3", "1200"),
    ("Final Answer: -2.50", "-2.5"),
]

for raw, expected in cases:
    actual = normalize_answer(raw)
    print(f"{raw!r} -> {actual!r}, expected={expected!r}")
    assert actual == expected

print("answer normalization tests passed")
assert answers_equivalent("Final Answer: 26.00", "26")
assert answers_equivalent("Final Answer: 50%", "0.5")
print("answer equivalence tests passed")
