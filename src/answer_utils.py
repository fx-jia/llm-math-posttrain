import re
from decimal import Decimal, InvalidOperation


def normalize_answer(text: str) -> str:
    text = str(text).strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.rstrip(".")

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        text = numbers[-1]

    try:
        value = Decimal(text)
    except InvalidOperation:
        return text.strip()

    if value == value.to_integral_value():
        return str(value.quantize(Decimal("1")))

    return format(value.normalize(), "f")
