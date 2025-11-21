# core/pool.py
import os
from core.TaskPoolManager import TaskPoolManager

_pool = None

def get_pool():
    global _pool
    if _pool is None:
        # 从环境变量读取并发度/队列大小/执行模式
        max_workers = int(os.getenv("POOL_MAX_WORKERS", "2"))
        max_queue_size = int(os.getenv("POOL_MAX_QUEUE", "1000"))
        use_process = os.getenv("POOL_USE_PROCESS", "1") not in ("0", "false", "False")
        _pool = TaskPoolManager(
            max_workers=max_workers,
            max_queue_size=max_queue_size,
            use_process=use_process
        )
    return _pool
