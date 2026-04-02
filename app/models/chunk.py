"""分块模型定义。"""

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class Chunk:
    """文档分块后的最小检索单元。

    Attributes:
        chunk_id: 全局唯一分块 ID。
        source: 分块来源文档标识。
        doc_type: 来源文档类型。
        chunk_index: 在同一文档中的分块序号（从 0 开始）。
        text: 分块文本内容。
        start_char: 在规范化文本中的起始字符下标。
        end_char: 在规范化文本中的结束字符下标（开区间右边界）。
        metadata: 继承自文档的扩展元信息。
    """

    chunk_id: str
    source: str
    doc_type: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, str] = field(default_factory=dict)
