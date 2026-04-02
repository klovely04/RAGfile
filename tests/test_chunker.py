"""分块服务测试。"""

import pytest

from app.models import Document
from app.services.chunker import ChunkingConfig, build_chunk_debug_view, chunk_document


def test_chunk_document_with_chinese_text() -> None:
    """中文文本应被切成多个分块，且长度受配置约束。"""
    document = Document(
        content="第一句。第二句比较长一些，需要参与分块处理。第三句继续补充上下文信息。第四句结束。",
        source="demo.txt",
        doc_type="txt",
        metadata={"file_name": "demo.txt"},
    )
    config = ChunkingConfig(chunk_size=20, chunk_overlap=5, min_chunk_size=8)

    chunks = chunk_document(document, config=config)

    assert len(chunks) >= 2
    assert all(len(chunk.text) <= config.chunk_size for chunk in chunks)
    assert chunks[0].text.startswith("第一句")


def test_chunk_document_empty_text_return_empty() -> None:
    """空文本应返回空分块列表。"""
    document = Document(content="   \n\t", source="empty.txt", doc_type="txt")

    chunks = chunk_document(document)

    assert chunks == []


def test_chunk_document_invalid_config_raise_error() -> None:
    """非法配置应抛出参数校验异常。"""
    document = Document(content="abc", source="demo.txt", doc_type="txt")

    with pytest.raises(ValueError, match="chunk_overlap 必须小于 chunk_size"):
        chunk_document(document, config=ChunkingConfig(chunk_size=10, chunk_overlap=10))


def test_build_chunk_debug_view() -> None:
    """调试视图应包含预期字段。"""
    document = Document(
        content="一句话。二句话。三句话。四句话。",
        source="debug.txt",
        doc_type="txt",
    )
    chunks = chunk_document(
        document,
        config=ChunkingConfig(chunk_size=8, chunk_overlap=2, min_chunk_size=4),
    )

    debug_rows = build_chunk_debug_view(chunks, preview_chars=4)

    assert len(debug_rows) == len(chunks)
    assert {"chunk_id", "chunk_index", "length", "span", "preview"} <= set(debug_rows[0].keys())
