"""数学答案归一化工具。

评测时只关心最终数值，因此需要消除货币符号、千分位和小数表示差异。
"""

import re
from decimal import Decimal, InvalidOperation


def normalize_answer(text: str) -> str:
    """提取文本中最后一个数字，并转成稳定的十进制字符串。

    例如 ``"$1,200.00."`` 会被归一化为 ``"1200"``。无法解析为
    Decimal 时保留清理后的原文，便于兼容非纯数字答案。
    """
    text = str(text).strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.rstrip(".")

    # 推理文本可能含有中间计算；约定最后出现的数值为答案。
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        text = numbers[-1]

    try:
        value = Decimal(text)
    except InvalidOperation:
        return text.strip()

    # 统一 1、1.0 和 1.00，避免等价答案被误判。
    if value == value.to_integral_value():
        return str(value.quantize(Decimal("1")))

    return format(value.normalize(), "f")
