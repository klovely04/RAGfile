"""对话结果模型定义。"""

from dataclasses import dataclass, field

from app.models.retrieval import RetrievalHit


@dataclass(slots=True, frozen=True)
class ChatResult:
    """一次问答请求的返回结果。

    Attributes:
        session_id: 会话 ID。
        answer: 模型最终回答文本。
        retrieved_chunks: 本次回答使用的检索证据列表。
    """

    session_id: str
    answer: str
    retrieved_chunks: list[RetrievalHit] = field(default_factory=list)
