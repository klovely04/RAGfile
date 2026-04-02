"""文档解析接口测试。"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_parse_document_api_txt_success() -> None:
    """txt 上传解析应成功。"""
    files = {"file": ("note.txt", b"hello rag", "text/plain")}

    resp = client.post("/api/v1/documents/parse", files=files)

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_type"] == "txt"
    assert body["content_length"] == 9
    assert body["content_preview"] == "hello rag"
    assert body["metadata"]["extension"] == ".txt"


def test_parse_document_api_unsupported_extension() -> None:
    """不支持扩展名应返回 400。"""
    files = {"file": ("sheet.xlsx", b"fake-data", "application/octet-stream")}

    resp = client.post("/api/v1/documents/parse", files=files)

    assert resp.status_code == 400
    assert "暂不支持的文件类型" in resp.json()["detail"]


def test_parse_document_api_empty_upload() -> None:
    """空文件上传应返回 400。"""
    files = {"file": ("empty.txt", b"", "text/plain")}

    resp = client.post("/api/v1/documents/parse", files=files)

    assert resp.status_code == 400
    assert resp.json()["detail"] == "上传文件为空"


def test_parse_document_api_no_extension() -> None:
    """无扩展名应返回 400。"""
    files = {"file": ("readme", b"hello", "text/plain")}

    resp = client.post("/api/v1/documents/parse", files=files)

    assert resp.status_code == 400
    assert resp.json()["detail"] == "文件缺少扩展名"
