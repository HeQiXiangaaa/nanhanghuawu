# core/logging_config.py
import logging

_initialized = getattr(logging, "_MY_APP_LOG_INITIALIZED", False)
if not _initialized:
    root = logging.getLogger()
    if not root.handlers:  # 防重入
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(processName)s] [%(name)s] %(message)s")
        # fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
        h.setFormatter(fmt)
        root.addHandler(h)
        root.setLevel(logging.INFO)
    logging._MY_APP_LOG_INITIALIZED = True  # ✅ 标记一次
    logging.getLogger(__name__).info("全局日志已初始化")


def setup_logging():
    """
    简单的全局控制台日志配置：
    - 不写任何日志文件
    - 只往控制台输出
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler（Flask debug 模式会重启多次）
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.info("全局日志已初始化（仅控制台，无全局日志文件）")
    return logger



