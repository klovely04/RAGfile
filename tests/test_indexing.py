"""索引与存储层测试。"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.models import Chunk, Document
from app.repositories import FaissStore, SQLiteStore
from app.services.index_service import IndexService


class FakeEmbeddingService:
    """用于测试索引流程的假向量服务。"""

    def encode_texts(self, texts: list[str]) -> NDArray[np.float32]:
        vectors = np.zeros((len(texts), 4), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row] = np.array(
                [len(text), row + 1, (row + 1) * 2, (row + 1) * 3],
                dtype=np.float32,
            )
        return vectors


def test_sqlite_store_upsert_and_query_chunks(tmp_path: Path) -> None:
    """SQLite 应支持文档+分块写入和读取。"""
    db_path = str(tmp_path / "rag.sqlite3")
    store = SQLiteStore(db_path)

    document = Document(
        content="第一段。第二段。",
        source="a.txt",
        doc_type="txt",
        metadata={"file_name": "a.txt"},
    )
    chunks = [
        Chunk(
            chunk_id="a-0",
            source="a.txt",
            doc_type="txt",
            chunk_index=0,
            text="第一段。",
            start_char=0,
            end_char=4,
            metadata={"file_name": "a.txt"},
        ),
        Chunk(
            chunk_id="a-1",
            source="a.txt",
            doc_type="txt",
            chunk_index=1,
            text="第二段。",
            start_char=4,
            end_char=8,
            metadata={"file_name": "a.txt"},
        ),
    ]

    store.upsert_document_with_chunks(document, chunks)
    loaded = store.list_all_chunks()

    assert len(loaded) == 2
    assert loaded[0].chunk_id == "a-0"
    assert loaded[1].text == "第二段。"


def test_index_service_rebuild_and_append(tmp_path: Path) -> None:
    """索引服务应支持全量重建与增量追加。"""
    db_path = str(tmp_path / "rag.sqlite3")
    index_path = str(tmp_path / "rag.faiss")
    sqlite_store = SQLiteStore(db_path)
    faiss_store = FaissStore(index_path)
    fake_embedding = FakeEmbeddingService()
    service = IndexService(
        sqlite_store=sqlite_store,
        faiss_store=faiss_store,
        embedding_service=fake_embedding,
    )

    document = Document(
        content="A",
        source="a.txt",
        doc_type="txt",
        metadata={},
    )
    chunks = [
        Chunk(
            chunk_id="a-0",
            source="a.txt",
            doc_type="txt",
            chunk_index=0,
            text="hello",
            start_char=0,
            end_char=5,
            metadata={},
        ),
        Chunk(
            chunk_id="a-1",
            source="a.txt",
            doc_type="txt",
            chunk_index=1,
            text="world",
            start_char=5,
            end_char=10,
            metadata={},
        ),
    ]
    sqlite_store.upsert_document_with_chunks(document, chunks)

    rebuilt = service.rebuild_from_store()
    assert rebuilt.chunk_count == 2
    assert rebuilt.vector_dimension == 4
    assert len(sqlite_store.list_index_chunk_ids()) == 2

    append_result = service.append_chunks(
        [
            sqlite_store.get_chunks_by_ids(["a-1"])[0],
        ]
    )
    assert append_result.mode == "append"
    assert len(sqlite_store.list_index_chunk_ids()) == 3
