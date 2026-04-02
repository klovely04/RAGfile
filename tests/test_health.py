"""健康检查接口测试。"""

from fastapi.testclient import TestClient

from app.main import app

# 测试客户端：用于模拟 HTTP 请求。
client = TestClient(app)


def test_health() -> None:
    """验证健康检查接口返回结构。"""
    # 调用健康检查接口。
    resp = client.get("/api/v1/health")
    # 状态码应为 200。
    assert resp.status_code == 200
    # 解析 JSON。
    body = resp.json()
    # status 字段固定为 ok。
    assert body["status"] == "ok"
    # 应包含时间戳字段。
    assert "timestamp" in body
