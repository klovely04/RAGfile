"""服务层导出入口。

服务层是业务逻辑核心，主要职责包括：
- 文档解析与文本切块；
- 向量化与索引构建；
- 向量检索与 RAG 对话编排。
"""

from app.services.chunker import ChunkingConfig, build_chunk_debug_view, chunk_document
from app.services.document_parser import DocumentParseError, parse_document
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.index_service import IndexBuildResult, IndexService, build_index_service
from app.services.llm_service import DeepSeekClient, LLMConfig, LLMServiceError
from app.services.rag_chat_service import ChatConfig, RAGChatService, build_chat_messages
from app.services.retriever_service import RetrieverService, build_retriever_service

# 对外公开的服务能力。
__all__ = [
    "parse_document",
    "DocumentParseError",
    "chunk_document",
    "ChunkingConfig",
    "build_chunk_debug_view",
    "EmbeddingService",
    "get_embedding_service",
    "IndexService",
    "IndexBuildResult",
    "build_index_service",
    "RetrieverService",
    "build_retriever_service",
    "DeepSeekClient",
    "LLMConfig",
    "LLMServiceError",
    "RAGChatService",
    "ChatConfig",
    "build_chat_messages",
]
