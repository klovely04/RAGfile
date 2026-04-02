"""文本分块服务。"""

from dataclasses import dataclass
from pathlib import Path

from app.models import Chunk, Document
from app.services.text_normalizer import normalize_text

# 常见中文/英文断句符与换行符，作为候选切分边界。
_SPLIT_PUNCTUATION = {"。", "，", "！", "!", "?", "？", ";", "；", "\n"}


@dataclass(slots=True, frozen=True)
class ChunkingConfig:
    """分块参数配置。

    Attributes:
        chunk_size: 每块目标最大长度（字符数）。
        chunk_overlap: 相邻块重叠字符数。
        min_chunk_size: 最小块长度，用于避免碎片化切分。
    """

    chunk_size: int = 500
    chunk_overlap: int = 80
    min_chunk_size: int = 50


def chunk_document(document: Document, config: ChunkingConfig | None = None) -> list[Chunk]:
    """把文档切分为可检索分块列表。

    Args:
        document: 待切分文档。
        config: 切分参数；若为空则使用默认配置。

    Returns:
        list[Chunk]: 生成的分块列表。
    """
    # 读取配置（无入参时走默认）。
    current_config = config or ChunkingConfig()
    # 先做参数合法性校验。
    _validate_config(current_config)

    # 先规范化文本，减少脏数据带来的切分误差。
    normalized_text = normalize_text(document.content)
    # 规范化后为空，直接返回空列表。
    if not normalized_text:
        return []

    # 预构建边界列表，提升后续切分效率。
    boundaries = _build_boundaries(normalized_text)
    chunks: list[Chunk] = []
    # 当前块起点下标。
    start = 0
    # 文本总长度。
    text_length = len(normalized_text)
    # source 的 basename 用于构造更直观的 chunk_id。
    source_name = Path(document.source).name or "document"

    # 滑动窗口式切分。
    while start < text_length:
        # 当前块目标终点（不超过文本末尾）。
        target_end = min(start + current_config.chunk_size, text_length)
        # 优先按断句边界确定真实终点。
        end = _pick_split_end(
            boundaries=boundaries,
            start=start,
            target_end=target_end,
            text_length=text_length,
            min_chunk_size=current_config.min_chunk_size,
        )
        # 防御性判断：若终点不前进，跳出循环避免死循环。
        if end <= start:
            break

        # 提取文本片段并去首尾空白。
        chunk_text = normalized_text[start:end].strip()
        if chunk_text:
            # 分块序号即当前列表长度。
            chunk_index = len(chunks)
            # 组装分块对象。
            chunks.append(
                Chunk(
                    chunk_id=f"{source_name}-{chunk_index}-{start}-{end}",
                    source=document.source,
                    doc_type=document.doc_type,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata=document.metadata,
                )
            )

        # 到达文本末尾则结束。
        if end >= text_length:
            break

        # 下一块起点向后挪动，但保留 overlap 区间。
        next_start = max(0, end - current_config.chunk_overlap)
        # 防止 overlap 配置不当导致起点不前进。
        start = end if next_start <= start else next_start

    return chunks


def build_chunk_debug_view(chunks: list[Chunk], preview_chars: int = 80) -> list[dict[str, object]]:
    """构建分块调试视图（给前端/日志快速查看）。

    Args:
        chunks: 分块列表。
        preview_chars: 预览文本长度。

    Returns:
        list[dict[str, object]]: 每块简要信息。
    """
    return [
        {
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "length": len(chunk.text),
            "span": [chunk.start_char, chunk.end_char],
            "preview": chunk.text[:preview_chars],
        }
        for chunk in chunks
    ]


def _validate_config(config: ChunkingConfig) -> None:
    """校验分块参数。

    Args:
        config: 待校验配置。
    """
    if config.chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if config.chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能为负数")
    if config.chunk_overlap >= config.chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    if config.min_chunk_size <= 0:
        raise ValueError("min_chunk_size 必须大于 0")
    if config.min_chunk_size > config.chunk_size:
        raise ValueError("min_chunk_size 不能大于 chunk_size")


def _build_boundaries(text: str) -> list[int]:
    """构建候选边界下标列表。

    Args:
        text: 规范化后的文本。

    Returns:
        list[int]: 包含 0 与文本末尾的边界下标列表。
    """
    boundaries = [0]
    for index, char in enumerate(text, start=1):
        if char in _SPLIT_PUNCTUATION:
            boundaries.append(index)
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    return boundaries


def _pick_split_end(
    *,
    boundaries: list[int],
    start: int,
    target_end: int,
    text_length: int,
    min_chunk_size: int,
) -> int:
    """在目标终点附近选择一个合理切分点。

    Args:
        boundaries: 候选边界下标列表。
        start: 当前块起点。
        target_end: 目标终点。
        text_length: 文本总长度。
        min_chunk_size: 最小块长度约束。

    Returns:
        int: 真实切分终点。
    """
    # 目标已到结尾时，直接返回文本末尾。
    if target_end >= text_length:
        return text_length

    # 计算最早允许边界（避免太短）。
    earliest = start + min_chunk_size
    candidate = start
    for boundary in boundaries:
        if boundary < earliest:
            continue
        if boundary <= target_end:
            candidate = boundary
        else:
            break

    # 找到候选边界就用候选，否则退回 target_end 硬切。
    if candidate > start:
        return candidate
    return target_end
