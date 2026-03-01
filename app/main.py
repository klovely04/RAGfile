import logging

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.core.config import get_settings
from app.core.logging import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(health_router, prefix=settings.api_prefix)


@app.on_event("startup")
def on_startup() -> None:
    logger.info("api_startup", extra={"env": settings.app_env})


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "RAG Knowledge Assistant API", "docs": "/docs"}
