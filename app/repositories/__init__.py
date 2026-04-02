"""仓储层导出入口。

仓储层负责“与存储系统交互”，主要包括：
- `SQLiteStore`：关系型持久化；
- `FaissStore`：向量索引持久化。
"""

from app.repositories.faiss_store import FaissStore
from app.repositories.sqlite_store import SQLiteStore, StoredChunk

# 对外公开的仓储类型。
__all__ = ["SQLiteStore", "StoredChunk", "FaissStore"]
