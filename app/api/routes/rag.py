"""RAG 主流程接口。"""

from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.dependencies import get_chat_service, get_index_service, get_sqlite_store
from app.models import Document
from app.services import (
    ChunkingConfig,
    DocumentParseError,
    build_chunk_debug_view,
    chunk_document,
    parse_document,
)

# RAG 路由分组。
router = APIRouter(tags=["rag"])
# 统一声明上传文件入参。
UPLOAD_FILE = File(...)


class ChatRequest(BaseModel):
    """聊天接口请求体。

    Attributes:
        question: 用户问题，长度限制 1~2000。
        session_id: 会话标识；为空时后端自动生成。
        source_filter: 可选来源过滤（只在指定来源里检索）。
    """

    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    source_filter: str | None = None


def _ok_response(data: object, message: str = "ok") -> dict[str, object]:
    """统一成功响应结构。

    Args:
        data: 业务数据主体。
        message: 提示信息，默认 `"ok"`。

    Returns:
        dict[str, object]: 统一格式 `{code, message, data}`。
    """
    return {"code": 0, "message": message, "data": data}


@router.post("/upload")
async def upload_document(file: UploadFile = UPLOAD_FILE) -> dict[str, object]:
    """上传文档并触发“解析 -> 切块 -> 入库 -> 重建索引”。

    Args:
        file: 上传文件对象。

    Returns:
        dict[str, object]: 包含分块结果和索引统计信息。

    Raises:
        HTTPException:
            - 400：文件校验或解析失败。
    """
    # 文件名不能为空。
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 提取扩展名。
    suffix = Path(file.filename).suffix.lower()
    # 扩展名不能为空。
    if not suffix:
        raise HTTPException(status_code=400, detail="文件缺少扩展名")

    # 读取上传字节内容。
    data = await file.read()
    # 空文件不处理。
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空")

    # 解析器要求输入路径，所以先写临时文件。
    tmp_path: Path | None = None
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        # 第一步：解析文档。
        parsed = parse_document(tmp_path)
        # 第二步：构造统一 Document 模型。
        document = Document(
            content=parsed.content,
            source=file.filename,
            doc_type=parsed.doc_type,
            metadata={
                **parsed.metadata,
                "original_file_name": file.filename,
            },
        )

        # 读取分块配置。
        settings = get_settings()
        config = ChunkingConfig(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_size=settings.min_chunk_size,
        )
        # 第三步：执行文本分块。
        chunks = chunk_document(document, config=config)

        # 第四步：写入 SQLite（文档 + 分块）。
        sqlite_store = get_sqlite_store()
        sqlite_store.upsert_document_with_chunks(document, chunks)

        # 第五步：从存储全量重建向量索引，确保索引与数据库一致。
        index_result = get_index_service().rebuild_from_store()
        return _ok_response(
            {
                "file_name": file.filename,
                "doc_type": document.doc_type,
                "chunk_count": len(chunks),
                "chunk_debug": build_chunk_debug_view(chunks[:5], preview_chars=60),
                "index": {
                    "mode": index_result.mode,
                    "chunk_count": index_result.chunk_count,
                    "vector_dimension": index_result.vector_dimension,
                },
            }
        )
    except DocumentParseError as exc:
        # 解析错误返回 HTTP 400。
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        # 清理临时文件。
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


@router.post("/index/rebuild")
def rebuild_index() -> dict[str, object]:
    """手动触发索引全量重建。

    Returns:
        dict[str, object]: 返回重建模式、分块数量、向量维度。
    """
    result = get_index_service().rebuild_from_store()
    return _ok_response(
        {
            "mode": result.mode,
            "chunk_count": result.chunk_count,
            "vector_dimension": result.vector_dimension,
        }
    )


@router.post("/chat")
def chat(payload: ChatRequest) -> dict[str, object]:
    """问答接口：检索增强后调用大模型。

    Args:
        payload: 聊天请求体。

    Returns:
        dict[str, object]: 会话 ID、答案、检索证据片段。
    """
    # 若调用方未传会话 ID，则按时间戳自动生成。
    session_id = payload.session_id or datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    # 调用服务层完成完整 RAG 链路。
    result = get_chat_service().chat(
        session_id=session_id,
        question=payload.question,
        source_filter=payload.source_filter,
    )
    # 将模型对象转为接口响应字典。
    return _ok_response(
        {
            "session_id": result.session_id,
            "answer": result.answer,
            "retrieved_chunks": [
                {
                    "chunk_id": hit.chunk_id,
                    "score": hit.score,
                    "source": hit.source,
                    "chunk_index": hit.chunk_index,
                    "reason": hit.reason,
                }
                for hit in result.retrieved_chunks
            ],
        }
    )


@router.get("/history/{session_id}")
def get_history(
    session_id: str,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, object]:
    """查询某个会话的历史消息。

    Args:
        session_id: 会话标识。
        limit: 最大返回条数（1~100）。

    Returns:
        dict[str, object]: 会话 ID 与消息列表。
    """
    history = get_sqlite_store().list_chat_history(session_id=session_id, limit=limit)
    return _ok_response({"session_id": session_id, "messages": history})
