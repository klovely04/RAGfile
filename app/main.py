"""FastAPI 应用启动入口。

这个文件做三件事：
1. 读取配置并初始化日志；
2. 创建 FastAPI 实例并挂载路由；
3. 提供根路由和启动事件钩子。
"""

import logging

from fastapi import FastAPI

from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.rag import router as rag_router
from app.core.config import get_settings
from app.core.logging import setup_logging

# 读取全局配置（使用了缓存，因此进程内只会创建一次）。
settings = get_settings()
# 按配置中的日志级别初始化日志输出。
setup_logging(settings.log_level)
# 获取当前模块专属日志器，方便定位日志来源。
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用对象。
app = FastAPI(
    # 文档页显示名称。
    title=settings.app_name,
    # 当前示例项目版本号。
    version="0.1.0",
)

# 注册健康检查路由。
app.include_router(health_router, prefix=settings.api_prefix)
# 注册文档解析路由。
app.include_router(documents_router, prefix=settings.api_prefix)
# 注册 RAG 主流程路由。
app.include_router(rag_router, prefix=settings.api_prefix)


@app.on_event("startup")
def on_startup() -> None:
    """应用启动后执行一次。

    Returns:
        None: 仅写启动日志，不返回业务数据。
    """
    # 输出结构化启动日志，记录运行环境（dev/test/prod）。
    logger.info("api_startup", extra={"env": settings.app_env})


@app.get("/")
def root() -> dict[str, str]:
    """根路径接口，用于快速确认服务在线。

    Returns:
        dict[str, str]:
            - `message`：服务欢迎信息；
            - `docs`：Swagger 文档路径。
    """
    return {"message": "RAG Knowledge Assistant API", "docs": "/docs"}
