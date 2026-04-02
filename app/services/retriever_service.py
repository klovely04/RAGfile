"""向量检索服务。"""

from typing import Protocol

import numpy as np

from app.models import RetrievalHit
from app.repositories.faiss_store import FaissStore
from app.repositories.sqlite_store import SQLiteStore
from app.services.embedding_service import get_embedding_service


class _QueryEmbedder(Protocol):
    """检索服务需要的最小查询向量化接口。"""

    def encode_query(self, query: str) -> np.ndarray:
        ...


class RetrieverService:
    """执行“查询向量化 -> FAISS 检索 -> 结果过滤 -> 命中组装”的服务。"""

    def __init__(
        self,
        *,
        sqlite_store: SQLiteStore,
        faiss_store: FaissStore,
        embedding_service: _QueryEmbedder,
    ) -> None:
        """初始化检索服务依赖。

        Args:
            sqlite_store: SQLite 仓储。
            faiss_store: FAISS 仓储。
            embedding_service: 查询向量化服务。
        """
        self.sqlite_store = sqlite_store
        self.faiss_store = faiss_store
        self.embedding_service = embedding_service

    def retrieve(
        self,
        *,
        query: str,
        top_k: int,
        source_filter: str | None = None,
        score_threshold: float = 0.0,
    ) -> list[RetrievalHit]:
        """执行检索并返回结构化命中结果。

        Args:
            query: 用户查询文本。
            top_k: 召回候选数量。
            source_filter: 可选来源过滤。
            score_threshold: 相似度阈值（低于阈值会被过滤）。

        Returns:
            list[RetrievalHit]: 命中结果列表。
        """
        # 加载向量索引；不存在则无结果。
        index = self.faiss_store.load()
        if index is None:
            return []

        # 读取 vector_id 对应的 chunk_id 顺序映射。
        chunk_ids = self.sqlite_store.list_index_chunk_ids()
        if not chunk_ids:
            return []

        # 编码查询向量并执行 top-k 检索。
        query_vector = self.embedding_service.encode_query(query)
        scores, indices = self.faiss_store.search(index, query_vector, top_k)

        # 第一轮过滤：过滤非法下标和低分。
        candidate_ids: list[str] = []
        candidate_scores: list[float] = []
        for score, idx in zip(scores.tolist(), indices.tolist(), strict=False):
            if idx < 0 or idx >= len(chunk_ids):
                continue
            if score < score_threshold:
                continue
            candidate_ids.append(chunk_ids[idx])
            candidate_scores.append(float(score))

        # 从 SQLite 取回分块详情。
        chunks = self.sqlite_store.get_chunks_by_ids(candidate_ids)

        # 第二轮过滤：来源过滤 + 命中结构组装。
        hits: list[RetrievalHit] = []
        for chunk, score in zip(chunks, candidate_scores, strict=False):
            if source_filter and chunk.source != source_filter:
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    score=round(score, 6),
                    source=chunk.source,
                    text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    reason=f"score={score:.4f} >= threshold={score_threshold:.4f}",
                    metadata=chunk.metadata,
                )
            )
        return hits


def build_retriever_service(sqlite_store: SQLiteStore, faiss_store: FaissStore) -> RetrieverService:
    """构建生产用检索服务实例。

    Args:
        sqlite_store: SQLite 仓储。
        faiss_store: FAISS 仓储。

    Returns:
        RetrieverService: 绑定全局 embedding 单例的检索服务。
    """
    return RetrieverService(
        sqlite_store=sqlite_store,
        faiss_store=faiss_store,
        embedding_service=get_embedding_service(),
    )
