"""检索服务测试。"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.models import Chunk, Document
from app.repositories import FaissStore, SQLiteStore
from app.services.index_service import IndexService
from app.services.retriever_service import RetrieverService


class FakeEmbeddingService:
    """为测试构造可控向量空间。"""

    def encode_texts(self, texts: list[str]) -> NDArray[np.float32]:
        vectors: list[list[float]] = []
        for text in texts:
            if "apple" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> NDArray[np.float32]:
        if "apple" in query:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)


def test_retrieve_topk_and_filter(tmp_path: Path) -> None:
    """检索应支持 top-k 与来源过滤。"""
    sqlite_store = SQLiteStore(str(tmp_path / "rag.sqlite3"))
    faiss_store = FaissStore(str(tmp_path / "rag.faiss"))
    embedding = FakeEmbeddingService()
    index_service = IndexService(
        sqlite_store=sqlite_store,
        faiss_store=faiss_store,
        embedding_service=embedding,
    )

    document = Document(content="x", source="fruit.txt", doc_type="txt")
    chunks = [
        Chunk(
            chunk_id="c-apple",
            source="fruit.txt",
            doc_type="txt",
            chunk_index=0,
            text="apple is sweet",
            start_char=0,
            end_char=14,
            metadata={"topic": "fruit"},
        ),
        Chunk(
            chunk_id="c-banana",
            source="fruit.txt",
            doc_type="txt",
            chunk_index=1,
            text="banana is yellow",
            start_char=15,
            end_char=31,
            metadata={"topic": "fruit"},
        ),
    ]
    sqlite_store.upsert_document_with_chunks(document, chunks)
    index_service.rebuild_from_store()

    retriever = RetrieverService(
        sqlite_store=sqlite_store,
        faiss_store=faiss_store,
        embedding_service=embedding,
    )

    hits = retriever.retrieve(query="tell me apple", top_k=2, score_threshold=0.1)
    assert len(hits) >= 1
    assert hits[0].chunk_id == "c-apple"
    assert "score=" in hits[0].reason

    filtered = retriever.retrieve(
        query="tell me apple",
        top_k=2,
        source_filter="other.txt",
        score_threshold=0.1,
    )
    assert filtered == []
