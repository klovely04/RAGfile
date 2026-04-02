"""文档模型定义。"""

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class Document:
    """统一文档模型。

    Attributes:
        content: 文档正文内容（纯文本）。
        source: 文档来源标识（通常是文件名或路径）。
        doc_type: 文档类型（如 `txt`、`md`、`pdf`、`docx`）。
        metadata: 文档补充信息（文件名、编码、扩展名等）。
    """

    content: str
    source: str
    doc_type: str
    metadata: dict[str, str] = field(default_factory=dict)
