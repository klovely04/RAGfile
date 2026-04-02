"""健康检查路由。"""

from datetime import UTC, datetime

from fastapi import APIRouter

# 创建路由对象，并打上 health 标签，方便在 Swagger 中分组。
router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """健康检查接口。

    Returns:
        dict[str, str]:
            - `status`: 固定返回 `"ok"`；
            - `timestamp`: 当前 UTC 时间（ISO8601 字符串）。
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
    }
