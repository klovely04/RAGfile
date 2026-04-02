"""文档解析相关接口。"""

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services import DocumentParseError, parse_document

# 文档路由分组。
router = APIRouter(tags=["documents"])
# 统一声明文件入参，便于复用与类型提示。
UPLOAD_FILE = File(...)


@router.post("/documents/parse")
async def parse_document_api(file: UploadFile = UPLOAD_FILE) -> dict[str, object]:
    """仅解析上传文档，不入库、不建索引。

    Args:
        file: 通过 multipart/form-data 上传的文件对象。

    Returns:
        dict[str, object]:
            包含文档类型、内容长度、内容预览和元信息。

    Raises:
        HTTPException:
            - 400：文件名为空、无扩展名、文件内容为空、解析失败。
    """
    # 校验文件名是否存在。
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 提取并规范化扩展名。
    suffix = Path(file.filename).suffix.lower()
    # 校验是否包含扩展名。
    if not suffix:
        raise HTTPException(status_code=400, detail="文件缺少扩展名")

    # 读取上传文件字节内容。
    data = await file.read()
    # 文件为空直接返回 400。
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空")

    # 解析器当前按“文件路径”工作，因此需要临时落盘。
    tmp_path: Path | None = None
    try:
        # 创建临时文件并写入上传字节。
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        # 调用服务层解析文件。
        doc = parse_document(tmp_path)
        # 返回轻量解析结果（给前端预览）。
        return {
            "doc_type": doc.doc_type,
            "source": doc.source,
            "content_length": len(doc.content),
            "content_preview": doc.content[:200],
            "metadata": doc.metadata,
        }
    except DocumentParseError as exc:
        # 统一转成 HTTP 400，detail 直接给出解析失败原因。
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        # 无论成功失败，都清理临时文件，避免磁盘残留。
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
