"""文本规范化工具。

分块前做轻量清洗，保证切块边界稳定、检索效果更可控。
"""

import re

# 常见不可见字符（零宽空格、BOM 等）。
_INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
# 行内连续空格与制表符折叠规则。
_INLINE_SPACE_RE = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    """规范化输入文本。

    Args:
        text: 原始文本。

    Returns:
        str: 清洗后的文本。
    """
    # 统一换行符到 `\n`。
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # 删除不可见字符。
    normalized = _INVISIBLE_RE.sub("", normalized)

    lines: list[str] = []
    # 逐行处理，清理行内多余空白。
    for raw_line in normalized.split("\n"):
        line = _INLINE_SPACE_RE.sub(" ", raw_line).strip()
        lines.append(line)

    # 合并回文本。
    joined = "\n".join(lines)
    # 折叠连续空行为单换行，避免出现大量空块。
    joined = re.sub(r"\n{2,}", "\n", joined)
    # 去除首尾空白后返回。
    return joined.strip()
