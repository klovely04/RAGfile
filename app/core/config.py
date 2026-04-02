"""应用配置定义与读取逻辑。

配置来源：
1. 系统环境变量；
2. 项目根目录下的 `.env` 文件。
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用运行时配置模型。

    该模型由 `pydantic-settings` 负责从环境变量自动装配。
    """

    # ===== 应用基础配置 =====
    app_name: str = "RAG Knowledge Assistant"
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    # ===== 向量化配置 =====
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_provider: str = "hash"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    embedding_normalize: bool = True

    # ===== 分块配置 =====
    chunk_size: int = 500
    chunk_overlap: int = 80
    min_chunk_size: int = 50

    # ===== 存储路径配置 =====
    data_dir: str = "data"
    sqlite_db_path: str = "data/rag.sqlite3"
    faiss_index_path: str = "data/faiss.index"

    # ===== 检索与会话配置 =====
    retrieval_top_k: int = 4
    retrieval_score_threshold: float = 0.2
    history_max_messages: int = 20

    # ===== 大模型配置（DeepSeek）=====
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2

    # 指定 `.env` 文件读取规则。
    model_config = SettingsConfigDict(
        # 默认读取项目根目录下 .env。
        env_file=".env",
        # 统一使用 UTF-8 编码，避免中文乱码。
        env_file_encoding="utf-8",
        # 变量名大小写不敏感（便于部署时兼容不同写法）。
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """获取全局唯一配置对象（进程内缓存）。

    Returns:
        Settings: 已加载环境变量后的配置实例。
    """
    return Settings()
