"""LLM 服务与 RAG 对话服务测试。"""

from pathlib import Path
from typing import Any

import httpx
import pytest

from app.models import RetrievalHit
from app.repositories import SQLiteStore
from app.services.llm_service import DeepSeekClient, LLMConfig, LLMServiceError, _extract_content
from app.services.rag_chat_service import ChatConfig, RAGChatService, build_chat_messages


def test_extract_content_success() -> None:
    """应能正确提取大模型返回文本。"""
    data: dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "content": " hello ",
                }
            }
        ]
    }
    assert _extract_content(data) == "hello"


def test_deepseek_complete_without_key_raise_error() -> None:
    """未配置 API Key 时应抛出异常。"""
    client = DeepSeekClient(
        LLMConfig(
            api_key="",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            timeout_seconds=10.0,
            max_retries=1,
        )
    )

    with pytest.raises(LLMServiceError, match="DEEPSEEK_API_KEY 未配置"):
        client.complete([{"role": "user", "content": "hi"}])


def test_deepseek_complete_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """首次超时后应按重试逻辑成功。"""

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "done"}}]}

    class FakeClient:
        call_count = 0

        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb
            return None

        def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> FakeResponse:
            del url, headers, json
            FakeClient.call_count += 1
            if FakeClient.call_count == 1:
                raise httpx.ReadTimeout("timeout")
            return FakeResponse()

    monkeypatch.setattr("app.services.llm_service.httpx.Client", FakeClient)

    client = DeepSeekClient(
        LLMConfig(
            api_key="x",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            timeout_seconds=10.0,
            max_retries=2,
        )
    )

    text = client.complete([{"role": "user", "content": "hi"}])
    assert text == "done"


class FakeRetriever:
    """固定返回一条检索命中，用于测试对话编排。"""

    def retrieve(
        self,
        *,
        query: str,
        top_k: int,
        source_filter: str | None = None,
        score_threshold: float = 0.0,
    ) -> list[RetrievalHit]:
        del query, top_k, source_filter, score_threshold
        return [
            RetrievalHit(
                chunk_id="c-1",
                score=0.8,
                source="note.txt",
                text="知识片段",
                chunk_index=0,
                reason="score ok",
            )
        ]


class FakeLLMClient:
    """固定返回回答文本。"""

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        del messages, temperature
        return "这是回答"


class FailingLLMClient:
    """始终抛错，用于测试兜底回答。"""

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        del messages, temperature
        raise LLMServiceError("boom")


def test_build_chat_messages_contains_context() -> None:
    """构造的用户消息中应包含检索片段。"""
    hits = [
        RetrievalHit(
            chunk_id="c-1",
            score=0.7,
            source="a.txt",
            text="片段A",
            chunk_index=0,
            reason="ok",
        )
    ]
    messages = build_chat_messages(question="什么是A", hits=hits, history=[])
    assert messages[-1]["role"] == "user"
    assert "片段A" in messages[-1]["content"]


def test_rag_chat_service_save_history(tmp_path: Path) -> None:
    """成功问答后应把用户与助手消息都落库。"""
    sqlite_store = SQLiteStore(str(tmp_path / "rag.sqlite3"))
    service = RAGChatService(
        retriever=FakeRetriever(),
        llm_client=FakeLLMClient(),
        sqlite_store=sqlite_store,
        config=ChatConfig(top_k=3, score_threshold=0.1, history_max_messages=10),
    )

    result = service.chat(session_id="s1", question="你好")
    history = sqlite_store.list_chat_history("s1", limit=10)

    assert result.answer == "这是回答"
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_rag_chat_service_llm_failure_fallback(tmp_path: Path) -> None:
    """LLM 调用失败时应返回兜底提示。"""
    sqlite_store = SQLiteStore(str(tmp_path / "rag.sqlite3"))
    service = RAGChatService(
        retriever=FakeRetriever(),
        llm_client=FailingLLMClient(),
        sqlite_store=sqlite_store,
        config=ChatConfig(top_k=3, score_threshold=0.1, history_max_messages=10),
    )

    result = service.chat(session_id="s2", question="你好")

    assert "暂时无法调用大模型服务" in result.answer
