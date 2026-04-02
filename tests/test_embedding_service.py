"""向量化服务测试。"""

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from app.services.embedding_service import EmbeddingService


class FakeEmbeddingModel:
    """用于测试的假向量模型。"""

    def __init__(self) -> None:
        self.call_count = 0

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> NDArray[np.floating]:
        """返回可预测向量，便于断言。"""
        del batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings
        self.call_count += 1
        return np.asarray(
            [[float(len(text)), float(index + 1)] for index, text in enumerate(sentences)],
            dtype=np.float32,
        )


def test_encode_texts_with_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """相同输入重复编码时应命中缓存。"""
    service = EmbeddingService(
        model_name="fake",
        provider="hash",
        batch_size=8,
        device="cpu",
        normalize_embeddings=True,
    )
    fake_model = FakeEmbeddingModel()
    monkeypatch.setattr(service, "_load_model", lambda: fake_model)

    first = service.encode_texts(["abc", "hello"])
    second = service.encode_texts(["abc", "hello"])

    assert first.shape == (2, 2)
    assert second.shape == (2, 2)
    assert fake_model.call_count == 1


def test_encode_query_return_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    """encode_query 应返回一维向量。"""
    service = EmbeddingService(
        model_name="fake",
        provider="hash",
        batch_size=8,
        device="cpu",
        normalize_embeddings=False,
    )
    fake_model = FakeEmbeddingModel()
    monkeypatch.setattr(service, "_load_model", lambda: fake_model)

    vector = service.encode_query("question")

    assert vector.shape == (2,)
    assert cast(float, vector[0]) == float(len("question"))


def test_encode_texts_empty_raise_error() -> None:
    """空输入应抛出 ValueError。"""
    service = EmbeddingService(
        model_name="fake",
        provider="hash",
        batch_size=8,
        device="cpu",
        normalize_embeddings=True,
    )

    with pytest.raises(ValueError, match="texts 不能为空"):
        service.encode_texts([])
