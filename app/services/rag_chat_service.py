"""RAG 对话编排服务。"""

from dataclasses import dataclass
from typing import Protocol

from app.models import ChatResult, RetrievalHit
from app.repositories.sqlite_store import SQLiteStore
from app.services.llm_service import LLMServiceError


class _Retriever(Protocol):
    """检索依赖协议。"""

    def retrieve(
        self,
        *,
        query: str,
        top_k: int,
        source_filter: str | None = None,
        score_threshold: float = 0.0,
    ) -> list[RetrievalHit]:
        ...


class _LLMClient(Protocol):
    """大模型依赖协议。"""

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        ...


@dataclass(slots=True, frozen=True)
class ChatConfig:
    """对话服务配置。

    Attributes:
        top_k: 每次检索返回候选数。
        score_threshold: 检索分数阈值。
        history_max_messages: 参与提示词构建的历史消息上限。
    """

    top_k: int
    score_threshold: float
    history_max_messages: int


class RAGChatService:
    """执行完整 RAG 问答链路。"""

    def __init__(
        self,
        *,
        retriever: _Retriever,
        llm_client: _LLMClient,
        sqlite_store: SQLiteStore,
        config: ChatConfig,
    ) -> None:
        """初始化对话服务依赖。

        Args:
            retriever: 检索服务。
            llm_client: 大模型客户端。
            sqlite_store: SQLite 仓储（用于会话历史读写）。
            config: 对话服务配置。
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.sqlite_store = sqlite_store
        self.config = config

    def chat(
        self,
        *,
        session_id: str,
        question: str,
        source_filter: str | None = None,
    ) -> ChatResult:
        """执行一次问答。

        Args:
            session_id: 会话 ID。
            question: 用户问题。
            source_filter: 可选来源过滤条件。

        Returns:
            ChatResult: 包含答案与检索证据。
        """
        # 1) 检索证据片段。
        hits = self.retriever.retrieve(
            query=question,
            top_k=self.config.top_k,
            source_filter=source_filter,
            score_threshold=self.config.score_threshold,
        )

        # 2) 拉取历史消息用于多轮上下文。
        history = self.sqlite_store.list_chat_history(
            session_id=session_id,
            limit=self.config.history_max_messages,
        )

        # 3) 构建大模型消息列表。
        messages = build_chat_messages(question=question, hits=hits, history=history)

        # 4) 调用大模型；失败时给出兜底提示。
        try:
            answer = self.llm_client.complete(messages)
        except LLMServiceError:
            answer = "暂时无法调用大模型服务，请检查 API Key 或稍后重试。"

        # 5) 保存问答消息到会话历史。
        self.sqlite_store.save_chat_message(session_id=session_id, role="user", content=question)
        self.sqlite_store.save_chat_message(session_id=session_id, role="assistant", content=answer)

        return ChatResult(session_id=session_id, answer=answer, retrieved_chunks=hits)


def build_chat_messages(
    *,
    question: str,
    hits: list[RetrievalHit],
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """构建大模型输入消息列表。

    Args:
        question: 用户问题。
        hits: 检索命中列表。
        history: 历史消息列表。

    Returns:
        list[dict[str, str]]: 大模型消息列表。
    """
    # 把检索结果渲染成可读上下文块。
    context_text = _build_context_text(hits)

    # 系统提示：约束模型回答行为。
    system_prompt = (
        "你是一名企业知识库问答助手。"
        "请优先依据提供的检索上下文回答问题。"
        "如果上下文不足，请明确说明信息不足，不要编造。"
    )

    # 用户提示：包含原问题和检索证据。
    user_prompt = f"问题：\n{question}\n\n检索上下文：\n{context_text}"

    # 按顺序拼装消息：system -> history -> current user。
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _build_context_text(hits: list[RetrievalHit]) -> str:
    """把检索命中渲染为上下文字符串。

    Args:
        hits: 检索命中列表。

    Returns:
        str: 可直接拼入提示词的上下文文本。
    """
    if not hits:
        return "【未检索到相关片段】"

    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{index}] source={hit.source} "
            f"chunk={hit.chunk_index} score={hit.score:.4f}\n{hit.text}"
        )
    return "\n\n".join(blocks)
