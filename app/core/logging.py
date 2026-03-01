import logging
import sys

from pythonjsonlogger.jsonlogger import JsonFormatter


def setup_logging(level: str) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
