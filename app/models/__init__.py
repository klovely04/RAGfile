"""领域模型导出入口。

新同学可以直接从这里导入核心数据结构，不必记住各子文件路径。
"""

from app.models.chat import ChatResult
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.retrieval import RetrievalHit

# 对外公开的模型集合。
__all__ = ["Document", "Chunk", "RetrievalHit", "ChatResult"]
