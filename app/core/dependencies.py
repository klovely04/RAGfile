"""依赖对象工厂。

本模块统一创建并缓存以下对象：
- SQLite 存储；
- FAISS 存储；
- 索引服务、检索服务；
- DeepSeek 客户端；
- RAG 对话服务。

这样做的好处：
1. 路由层不用关心对象如何构建；
2. 避免重复初始化重对象；
3. 测试时可通过 monkeypatch 方便替换。
"""

from functools import lru_cache

from app.core.config import get_settings
from app.repositories import FaissStore, SQLiteStore
from app.services import (
    ChatConfig,
    DeepSeekClient,
    IndexService,
    LLMConfig,
    RAGChatService,
    RetrieverService,
    build_index_service,
    build_retriever_service,
)


@lru_cache
def get_sqlite_store() -> SQLiteStore:
    """获取 SQLite 存储单例。

    Returns:
        SQLiteStore: 用于文档、分块、聊天历史的持久化。
    """
    settings = get_settings()
    return SQLiteStore(settings.sqlite_db_path)


@lru_cache
def get_faiss_store() -> FaissStore:
    """获取 FAISS 存储单例。

    Returns:
        FaissStore: 用于向量索引持久化与检索。
    """
    settings = get_settings()
    return FaissStore(settings.faiss_index_path)


@lru_cache
def get_index_service() -> IndexService:
    """获取索引服务单例。

    Returns:
        IndexService: 负责全量重建/增量追加向量索引。
    """
    return build_index_service(get_sqlite_store(), get_faiss_store())


@lru_cache
def get_retriever_service() -> RetrieverService:
    """获取检索服务单例。

    Returns:
        RetrieverService: 负责向量召回与过滤。
    """
    return build_retriever_service(get_sqlite_store(), get_faiss_store())


@lru_cache
def get_llm_client() -> DeepSeekClient:
    """获取 DeepSeek 客户端单例。

    Returns:
        DeepSeekClient: 封装 DeepSeek API 调用与重试逻辑。
    """
    settings = get_settings()
    # 从配置拼装 LLM 调用参数。
    config = LLMConfig(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    return DeepSeekClient(config)


@lru_cache
def get_chat_service() -> RAGChatService:
    """获取 RAG 对话服务单例。

    Returns:
        RAGChatService: 完整执行“检索 + 提示词构建 + 大模型回答 + 历史落库”。
    """
    settings = get_settings()
    return RAGChatService(
        retriever=get_retriever_service(),
        llm_client=get_llm_client(),
        sqlite_store=get_sqlite_store(),
        config=ChatConfig(
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
            history_max_messages=settings.history_max_messages,
        ),
    )
