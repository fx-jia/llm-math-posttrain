import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.answer_utils import normalize_answer


cases = [
    ("26.00", "26"),
    ("88.00", "88"),
    ("$1,430", "1430"),
    ("Final Answer: 72.", "72"),
    ("5.94", "5.94"),
    ("14.67", "14.67"),
]

for raw, expected in cases:
    actual = normalize_answer(raw)
    print(f"{raw!r} -> {actual!r}, expected={expected!r}")
    assert actual == expected

print("answer normalization tests passed")
