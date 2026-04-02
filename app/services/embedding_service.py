"""向量化服务。

支持两种提供方式：
1. `sentence_transformers`：真实语义向量；
2. `hash`：哈希兜底向量（保证无模型依赖时流程可运行）。
"""

import logging
from functools import lru_cache
from hashlib import sha256
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.core.config import get_settings

# 当前模块日志器。
logger = logging.getLogger(__name__)

# 二维向量矩阵类型别名。
FloatMatrix = NDArray[np.float32]
# 一维向量类型别名。
FloatVector = NDArray[np.float32]


class _HashEmbeddingModel:
    """哈希兜底向量模型。

    这个模型不依赖第三方大模型文件，适合本地快速跑通流程。
    """

    def __init__(self, *, dimension: int = 384) -> None:
        """初始化哈希向量维度。

        Args:
            dimension: 生成向量维度，默认 384。
        """
        self.dimension = dimension

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> FloatMatrix:
        """把文本列表编码为可重复的伪向量。

        Args:
            sentences: 待编码文本列表。
            batch_size: 与 sentence-transformers 接口对齐，兜底模型不使用。
            show_progress_bar: 同上，不使用。
            convert_to_numpy: 同上，不使用。
            normalize_embeddings: 是否做 L2 归一化。

        Returns:
            FloatMatrix: 形状为 `(len(sentences), dimension)` 的向量矩阵。
        """
        # 兜底实现不需要这些参数，仅保持签名兼容。
        del batch_size, show_progress_bar, convert_to_numpy

        vectors = np.zeros((len(sentences), self.dimension), dtype=np.float32)
        for row, sentence in enumerate(sentences):
            # 用 SHA-256 生成稳定字节序列。
            digest = sha256(sentence.encode("utf-8")).digest()
            digest_bytes = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            # 重复填充到目标维度。
            tiled = np.resize(digest_bytes, self.dimension)
            vector = tiled / 255.0
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            vectors[row] = vector
        return vectors


class EmbeddingService:
    """统一向量化服务封装。"""

    def __init__(
        self,
        *,
        model_name: str,
        provider: str,
        batch_size: int,
        device: str,
        normalize_embeddings: bool,
    ) -> None:
        """初始化向量化服务。

        Args:
            model_name: 模型名称（provider 为 sentence_transformers 时生效）。
            provider: 向量提供方式（`hash` 或 `sentence_transformers`）。
            batch_size: 批量编码大小。
            device: 运行设备（`cpu`/`cuda`）。
            normalize_embeddings: 是否归一化输出向量。
        """
        self.model_name = model_name
        self.provider = provider
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        # 懒加载模型实例。
        self._model: Any | None = None
        # 简单内存缓存：相同文本重复请求时直接复用向量。
        self._cache: dict[str, FloatVector] = {}

    def encode_query(self, query: str) -> FloatVector:
        """编码单条查询文本。

        Args:
            query: 用户问题文本。

        Returns:
            FloatVector: 一维查询向量。
        """
        vectors = self.encode_texts([query])
        return np.asarray(vectors[0], dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> FloatMatrix:
        """批量编码文本。

        Args:
            texts: 待编码文本列表。

        Returns:
            FloatMatrix: 编码后的二维向量矩阵。
        """
        if not texts:
            raise ValueError("texts 不能为空")

        start = perf_counter()
        model = self._get_model()

        # 只编码缓存中没有的文本。
        uncached_texts = [text for text in texts if text not in self._cache]
        if uncached_texts:
            encoded = model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
            encoded_matrix = np.asarray(encoded, dtype=np.float32)
            for index, text in enumerate(uncached_texts):
                self._cache[text] = encoded_matrix[index]

        # 按原输入顺序拼回向量。
        vectors = np.vstack([self._cache[text] for text in texts]).astype(np.float32)
        elapsed_ms = (perf_counter() - start) * 1000
        logger.info(
            "embedding_encode_done",
            extra={
                "model_name": self.model_name,
                "text_count": len(texts),
                "cache_hit": len(texts) - len(uncached_texts),
                "latency_ms": round(elapsed_ms, 2),
            },
        )
        return vectors

    def _get_model(self) -> Any:
        """获取底层模型实例（懒加载）。"""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        """加载底层向量模型。

        Returns:
            Any: sentence-transformers 模型实例或哈希兜底实例。
        """
        logger.info(
            "embedding_model_load_start",
            extra={
                "model_name": self.model_name,
                "provider": self.provider,
                "device": self.device,
            },
        )
        # 非 sentence_transformers 直接走哈希兜底。
        if self.provider != "sentence_transformers":
            logger.info("embedding_model_load_hash_provider", extra={"provider": self.provider})
            return _HashEmbeddingModel()

        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("embedding_model_load_done", extra={"model_name": self.model_name})
            return model
        except Exception as exc:
            # 真实模型加载失败时自动降级。
            logger.warning(
                "embedding_model_load_fallback",
                extra={"model_name": self.model_name, "error": str(exc)},
            )
            return _HashEmbeddingModel()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """获取全局向量服务单例。

    Returns:
        EmbeddingService: 按配置构建的向量服务。
    """
    settings = get_settings()
    return EmbeddingService(
        model_name=settings.embedding_model_name,
        provider=settings.embedding_provider,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
        normalize_embeddings=settings.embedding_normalize,
    )
