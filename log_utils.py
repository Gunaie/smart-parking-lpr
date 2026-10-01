# -*- coding: utf-8 -*-
"""统一日志配置：控制台 + 文件轮转，支持请求 ID（contextvars）。

用法：
    from log_utils import get_logger, setup_logging
    setup_logging()
    logger = get_logger(__name__)
    logger.info("消息")
"""
import contextvars
import logging
import uuid
from logging.handlers import RotatingFileHandler

from config import settings

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s]: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_initialized = False

# 请求 ID 上下文变量
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-")


class _RequestIdFilter(logging.Filter):
    """从 contextvars 读取 request_id 注入日志。"""

    def filter(self, record):
        record.request_id = _request_id_var.get()
        return True


def set_request_id(rid: str):
    """设置当前请求 ID（中间件调用）。"""
    _request_id_var.set(rid)


def setup_logging(level: int = logging.INFO) -> None:
    """初始化全局日志配置（幂等）。"""
    global _initialized
    if _initialized:
        return

    log_dir = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)
    req_filter = _RequestIdFilter()

    # 控制台
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(req_filter)

    # 文件轮转：单文件 5MB，保留 5 个
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(req_filter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # 降低第三方库日志噪音
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)


def new_request_id() -> str:
    """生成短请求 ID。"""
    return uuid.uuid4().hex[:8]
