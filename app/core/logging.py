"""日志初始化工具。"""

import logging
import sys

from pythonjsonlogger.jsonlogger import JsonFormatter


def setup_logging(level: str) -> None:
    """初始化根日志器为 JSON 输出。

    Args:
        level: 日志级别字符串，例如 `"INFO"`、`"DEBUG"`。

    Returns:
        None: 该函数只修改全局日志配置。
    """
    # 获取根日志器（全局共享）。
    root_logger = logging.getLogger()
    # 设置日志级别，统一转大写保证兼容。
    root_logger.setLevel(level.upper())

    # 创建标准输出处理器，适合本地和容器日志采集。
    handler = logging.StreamHandler(sys.stdout)
    # 使用 JSON 格式器，方便后续检索和结构化分析。
    formatter = JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)

    # 清空旧处理器，避免重复打印日志。
    root_logger.handlers.clear()
    # 挂载当前处理器。
    root_logger.addHandler(handler)
