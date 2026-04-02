"""检索命中模型定义。"""

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class RetrievalHit:
    """一次检索命中的结构化结果。

    Attributes:
        chunk_id: 命中的分块 ID。
        score: 相似度分数（越大越相关）。
        source: 命中文档来源。
        text: 命中的分块文本。
        chunk_index: 命中分块在文档中的序号。
        reason: 命中原因说明（用于可解释输出）。
        metadata: 命中分块的扩展元信息。
    """

    chunk_id: str
    score: float
    source: str
    text: str
    chunk_index: int
    reason: str
    metadata: dict[str, str] = field(default_factory=dict)
