"""RAG 路由集成测试。"""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from app.main import app
from app.repositories import FaissStore, SQLiteStore
from app.services.index_service import IndexService
from app.services.rag_chat_service import ChatConfig, RAGChatService
from app.services.retriever_service import RetrieverService


class FakeEmbeddingService:
    """用于接口测试的假向量服务。"""

    def encode_texts(self, texts: list[str]) -> NDArray[np.float32]:
        vectors = np.zeros((len(texts), 4), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row] = np.array([len(text), row + 1, 1.0, 1.0], dtype=np.float32)
        return vectors

    def encode_query(self, query: str) -> NDArray[np.float32]:
        return np.array([len(query), 1.0, 1.0, 1.0], dtype=np.float32)


class FakeLLM:
    """用于接口测试的假大模型。"""

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        del messages, temperature
        return "这是测试回答"


@pytest.fixture
def rag_test_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """构造隔离环境下的测试客户端。"""
    sqlite_store = SQLiteStore(str(tmp_path / "rag.sqlite3"))
    faiss_store = FaissStore(str(tmp_path / "rag.faiss"))
    embedding = FakeEmbeddingService()
    index_service = IndexService(
        sqlite_store=sqlite_store,
        faiss_store=faiss_store,
        embedding_service=embedding,
    )
    retriever = RetrieverService(
        sqlite_store=sqlite_store,
        faiss_store=faiss_store,
        embedding_service=embedding,
    )
    chat_service = RAGChatService(
        retriever=retriever,
        llm_client=FakeLLM(),
        sqlite_store=sqlite_store,
        config=ChatConfig(top_k=3, score_threshold=0.0, history_max_messages=20),
    )

    # 把真实依赖替换为测试依赖，避免污染本地数据。
    monkeypatch.setattr("app.api.routes.rag.get_sqlite_store", lambda: sqlite_store)
    monkeypatch.setattr("app.api.routes.rag.get_index_service", lambda: index_service)
    monkeypatch.setattr("app.api.routes.rag.get_chat_service", lambda: chat_service)

    return TestClient(app)


def test_upload_rebuild_chat_and_history(rag_test_client: TestClient) -> None:
    """验证上传、重建、问答、历史查询完整链路。"""
    upload_resp = rag_test_client.post(
        "/api/v1/upload",
        files={"file": ("demo.txt", b"RAG system test text.", "text/plain")},
    )
    assert upload_resp.status_code == 200
    upload_body = upload_resp.json()
    assert upload_body["code"] == 0
    assert upload_body["data"]["chunk_count"] >= 1

    rebuild_resp = rag_test_client.post("/api/v1/index/rebuild")
    assert rebuild_resp.status_code == 200
    assert rebuild_resp.json()["code"] == 0

    chat_resp = rag_test_client.post(
        "/api/v1/chat",
        json={"session_id": "s-test", "question": "这份文档讲什么？"},
    )
    assert chat_resp.status_code == 200
    chat_body = chat_resp.json()
    assert chat_body["code"] == 0
    assert chat_body["data"]["session_id"] == "s-test"
    assert chat_body["data"]["answer"] == "这是测试回答"

    history_resp = rag_test_client.get("/api/v1/history/s-test")
    assert history_resp.status_code == 200
    history_body = history_resp.json()
    assert history_body["code"] == 0
    assert len(history_body["data"]["messages"]) == 2
