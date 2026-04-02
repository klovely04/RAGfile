"""索引构建服务。"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.repositories.faiss_store import FaissStore
from app.repositories.sqlite_store import SQLiteStore, StoredChunk
from app.services.embedding_service import get_embedding_service


@dataclass(slots=True, frozen=True)
class IndexBuildResult:
    """索引构建结果摘要。"""

    chunk_count: int
    vector_dimension: int
    mode: str


class _Embedder(Protocol):
    """索引服务需要的最小向量化接口。"""

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        ...


class IndexService:
    """负责把 SQLite 分块同步为 FAISS 向量索引。"""

    def __init__(
        self,
        *,
        sqlite_store: SQLiteStore,
        faiss_store: FaissStore,
        embedding_service: _Embedder,
    ) -> None:
        """初始化索引服务依赖。

        Args:
            sqlite_store: SQLite 仓储对象。
            faiss_store: FAISS 仓储对象。
            embedding_service: 向量化服务。
        """
        self.sqlite_store = sqlite_store
        self.faiss_store = faiss_store
        self.embedding_service = embedding_service

    def rebuild_from_store(self) -> IndexBuildResult:
        """从 SQLite 全量重建向量索引。

        Returns:
            IndexBuildResult: 构建结果统计信息。
        """
        chunks = self.sqlite_store.list_all_chunks()
        # 没有分块时，清空映射并返回空结果。
        if not chunks:
            self.sqlite_store.replace_index_entries([])
            return IndexBuildResult(chunk_count=0, vector_dimension=0, mode="rebuild")

        # 向量化全部分块文本。
        vectors = self._encode_chunks(chunks)
        # 重建 FAISS 索引。
        self.faiss_store.rebuild(vectors)
        # 重建 vector_id -> chunk_id 映射。
        self.sqlite_store.replace_index_entries([chunk.chunk_id for chunk in chunks])
        return IndexBuildResult(
            chunk_count=len(chunks),
            vector_dimension=int(vectors.shape[1]),
            mode="rebuild",
        )

    def append_chunks(self, chunks: list[StoredChunk]) -> IndexBuildResult:
        """向现有索引追加分块。

        Args:
            chunks: 需要追加的分块列表。

        Returns:
            IndexBuildResult: 追加结果统计信息。
        """
        if not chunks:
            return IndexBuildResult(chunk_count=0, vector_dimension=0, mode="append")

        # 向量化追加分块。
        vectors = self._encode_chunks(chunks)
        # 追加到 FAISS。
        self.faiss_store.append(vectors)
        # 追加到 SQLite 映射表。
        self.sqlite_store.append_index_entries([chunk.chunk_id for chunk in chunks])
        return IndexBuildResult(
            chunk_count=len(chunks),
            vector_dimension=int(vectors.shape[1]),
            mode="append",
        )

    def _encode_chunks(self, chunks: list[StoredChunk]) -> np.ndarray:
        """提取分块文本并执行向量化。

        Args:
            chunks: 分块列表。

        Returns:
            np.ndarray: 二维向量矩阵。
        """
        texts = [chunk.text for chunk in chunks]
        return self.embedding_service.encode_texts(texts)


def build_index_service(sqlite_store: SQLiteStore, faiss_store: FaissStore) -> IndexService:
    """构建生产用索引服务实例。

    Args:
        sqlite_store: SQLite 仓储。
        faiss_store: FAISS 仓储。

    Returns:
        IndexService: 绑定全局 embedding 单例的索引服务。
    """
    return IndexService(
        sqlite_store=sqlite_store,
        faiss_store=faiss_store,
        embedding_service=get_embedding_service(),
    )
