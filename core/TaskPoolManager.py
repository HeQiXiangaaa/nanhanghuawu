# core/TaskPoolManager.py
import time
import uuid
import json
import atexit
import logging
from dataclasses import dataclass, field
from pathlib import Path
from queue import PriorityQueue, Empty
from threading import Event, Thread, Lock
from typing import Any, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, Future, CancelledError, TimeoutError

from core.orchestrator import run_pipeline  # 复用你已有的单任务执行逻辑

logger = logging.getLogger(__name__)


# ======= 任务状态机 =======
PENDING   = "PENDING"    # 在等待队列
RUNNING   = "RUNNING"    # 已被执行器接收
SUCCEEDED = "SUCCEEDED"  # 任务成功
FAILED    = "FAILED"     # 任务失败
CANCELLED = "CANCELLED"  # 被取消


@dataclass(order=True)
class _QueueItem:
    priority: int
    enq_time: float
    task_id: str = field(compare=False)


@dataclass
class TaskRecord:
    task_id: str
    payload: Dict[str, Any]              # 即原来 /run 的请求体
    status: str = PENDING
    priority: int = 0
    retries: int = 0
    max_retries: int = 0
    timeout: Optional[int] = None        # 单任务超时时间（秒）
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    future: Optional[Future] = None

class TaskPoolManager:
    """
    多任务调度中枢：
      - 等待队列（可选优先级）
      - 限制最大并发（Process/Thread 池）
      - 生命周期管理（提交→执行→完成/失败/取消）
      - 指标/观测（队列长度、并发、QPS、平均耗时）
    """

    def __init__(
        self,
        max_workers: int = 2,
        max_queue_size: int = 1000,
        use_process: bool = True,
        poll_interval: float = 0.2,
    ):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.poll_interval = poll_interval

        self._queue: PriorityQueue[_QueueItem] = PriorityQueue()
        self._records: Dict[str, TaskRecord] = {}
        self._lock = Lock()
        self._stop = Event()
        self._dispatcher = Thread(target=self._dispatch_loop, daemon=True)

        # 执行器：默认进程池（更适合 CPU 密集训练）
        if use_process:
            self._executor = ProcessPoolExecutor(max_workers=max_workers)
            self._mode = "process"
        else:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            self._mode = "thread"

        self._running = 0
        self._completed = 0
        self._succeeded = 0
        self._failed = 0
        self._cancelled = 0
        self._total_time = 0.0

        self._dispatcher.start()
        atexit.register(self.shutdown)

        logger.info("[TaskPool] init: workers=%s mode=%s queue=%s",
                    max_workers, self._mode, max_queue_size)

    # ---------- 公共 API ----------

    def submit(
        self,
        payload: Dict[str, Any],
        priority: int = 0,
        timeout: Optional[int] = None,
        max_retries: int = 0,
    ) -> str:
        """
        提交一个任务（立即返回 task_id）
        payload: 即原先 /run 的 JSON（包含 task_type/dataset/model_name/...）
        """
        with self._lock:
            if self._queue.qsize() >= self.max_queue_size:
                raise RuntimeError("队列已满")

            task_id = uuid.uuid4().hex[:8]
            rec = TaskRecord(
                task_id=task_id,
                payload=payload,
                priority=int(priority),
                timeout=timeout,
                max_retries=max_retries,
            )
            self._records[task_id] = rec
            self._queue.put(_QueueItem(priority=priority, enq_time=time.time(), task_id=task_id))

        logger.info("[TaskPool] queued task=%s priority=%s", task_id, priority)
        return task_id

    def status(self, task_id: str) -> Dict[str, Any]:
        rec = self._records.get(task_id)
        if not rec:
            raise KeyError(f"未知任务: {task_id}")
        return {
            "task_id": task_id,
            "status": rec.status,
            "priority": rec.priority,
            "retries": rec.retries,
            "max_retries": rec.max_retries,
            "timeout": rec.timeout,
            "submitted_at": rec.submitted_at,
            "started_at": rec.started_at,
            "finished_at": rec.finished_at,
            "running": self._running,
            "queue_size": self._queue.qsize(),
            "mode": self._mode,
        }

    def result(self, task_id: str) -> Dict[str, Any]:
        rec = self._records.get(task_id)
        if not rec:
            raise KeyError(f"未知任务: {task_id}")
        if rec.status == SUCCEEDED:
            return rec.result or {}
        if rec.status in (FAILED, CANCELLED):
            return {"error": rec.error, "status": rec.status}
        # 仍在运行/排队
        return {"status": rec.status}

    def cancel(self, task_id: str) -> bool:
        rec = self._records.get(task_id)
        if not rec:
            return False
        # 仅能取消 PENDING 或尚未真正开始的 RUNNING（best-effort）
        if rec.status == PENDING:
            rec.status = CANCELLED
            self._cancelled += 1
            return True
        if rec.status == RUNNING and rec.future:
            ok = rec.future.cancel()
            if ok:
                rec.status = CANCELLED
                self._cancelled += 1
            return ok
        return False

    def metrics(self) -> Dict[str, Any]:
        avg_time = (self._total_time / self._completed) if self._completed else 0.0
        return {
            "workers": self.max_workers,
            "mode": self._mode,
            "queue_size": self._queue.qsize(),
            "running": self._running,
            "completed": self._completed,
            "succeeded": self._succeeded,
            "failed": self._failed,
            "cancelled": self._cancelled,
            "avg_elapsed_sec": round(avg_time, 3),
        }

    def shutdown(self):
        if self._stop.is_set():
            return
        logger.info("[TaskPool] shutting down dispatcher/executor ...")
        self._stop.set()
        try:
            self._dispatcher.join(timeout=2)
        except Exception:
            pass
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    # ---------- 内部：调度主循环 ----------

    def _dispatch_loop(self):
        while not self._stop.is_set():
            try:
                # 至少阻塞拿 1 个，避免空转
                first = self._queue.get(timeout=self.poll_interval)
            except Empty:
                continue

            batch = [first]
            # 计算空槽：最多发这么多
            available = max(self.max_workers - self._running - len(batch), 0)

            # 尽量把当下队列里的更高优先级任务“一把抓”出来
            for _ in range(available):
                try:
                    batch.append(self._queue.get_nowait())
                except Empty:
                    break

            # 统一派发（此刻 batch 已经按 (priority, enq_time) 排好）
            for item in batch:
                tid = item.task_id
                rec = self._records.get(tid)
                if not rec or rec.status == CANCELLED:
                    continue
                rec.status = RUNNING
                rec.started_at = time.time()
                self._running += 1
                payload2 = dict(rec.payload)
                payload2["force_job_id"] = rec.task_id  # 用 task_id 作为 job_id
                rec.future = self._executor.submit(_exec_one_task, payload2)
                # rec.future = self._executor.submit(_exec_one_task, rec.payload)
                rec.future.add_done_callback(lambda fut, t=tid: self._on_done(t, fut))

    def _on_done(self, task_id: str, fut: Future):
        rec = self._records.get(task_id)
        self._running -= 1
        self._completed += 1
        rec.finished_at = time.time()
        elapsed = rec.finished_at - (rec.started_at or rec.submitted_at)
        self._total_time += elapsed

        try:
            # 支持超时（如果配置了超时）
            if rec.timeout:
                result = fut.result(timeout=0)  # 回调触发时已完成；此处不再等待
            result = fut.result()
            rec.result = result
            rec.status = SUCCEEDED
            self._succeeded += 1
            logger.info("[TaskPool] task=%s OK in %.2fs", task_id, elapsed)
        except CancelledError:
            rec.status = CANCELLED
            self._cancelled += 1
            rec.error = "cancelled"
            logger.warning("[TaskPool] task=%s cancelled", task_id)
        except Exception as e:
            rec.error = repr(e)
            if rec.retries < rec.max_retries:
                # 重新入队（指数退避可自行加）
                rec.retries += 1
                rec.status = PENDING
                rec.started_at = None
                rec.finished_at = None
                self._queue.put(_QueueItem(priority=rec.priority, enq_time=time.time(), task_id=task_id))
                logger.warning("[TaskPool] task=%s failed, retry=%s/%s: %s",
                               task_id, rec.retries, rec.max_retries, rec.error)
            else:
                rec.status = FAILED
                self._failed += 1
                logger.exception("[TaskPool] task=%s failed: %s", task_id, rec.error)

# 在 TaskPoolManager 类末尾新增（非必须，但便于 app.py 调用）
    def submit_many(self, jobs, default_timeout=None, default_retries=0):
        """
        jobs: 迭代器，每个元素是单个任务 payload(dict)，可包含 priority/timeout/max_retries
        返回: [(task_id, payload), ...]
        """
        results = []
        for job in jobs:
            payload = dict(job)
            priority = int(payload.pop("priority", 0))
            timeout = payload.pop("timeout", default_timeout)
            max_retries = int(payload.pop("max_retries", default_retries or 0))
            tid = self.submit(payload, priority=priority, timeout=timeout, max_retries=max_retries)
            results.append((tid, job))
        return results

    def status_many(self, task_ids):
        return {tid: self.status(tid) for tid in task_ids if tid in self._records}

    def cancel_many(self, task_ids):
        res = {}
        for tid in task_ids:
            try:
                res[tid] = self.cancel(tid)
            except Exception as e:
                res[tid] = {"cancelled": False, "error": repr(e)}
        return res

# ---- TaskPoolManager 内新增：队列安全长度 ----
def _qsize_safe(self):
    try:
        return int(self._queue.qsize())
    except Exception:
        # 某些自定义队列可能没有 qsize()
        try:
            return len(self._queue)  # 退化
        except Exception:
            return None

# ---- TaskPoolManager 内新增：从 _records 推断状态 ----
def _aggregate_from_records(self):
    recs = getattr(self, "_records", {}) or {}
    total = len(recs)
    running = sum(1 for r in recs.values() if str(r.get("state","")).lower() in ("running","executing","in_progress"))
    completed = sum(1 for r in recs.values() if str(r.get("state","")).lower() in ("done","completed","success","finished"))
    failed = sum(1 for r in recs.values() if str(r.get("state","")).lower() in ("failed","error","cancelled","canceled"))
    pending = sum(1 for r in recs.values() if str(r.get("state","")).lower() in ("queued","pending","waiting"))
    return total, running, completed, failed, pending

# ---- TaskPoolManager 内新增：从 futures 推断状态 ----
def _aggregate_from_futures(self):
    futs = getattr(self, "_futures", {}) or {}
    total = len(futs)
    running = sum(1 for f in futs.values() if not f.done())
    completed = sum(1 for f in futs.values() if f.done() and (not f.cancelled()) and (f.exception() is None))
    failed = sum(1 for f in futs.values() if f.done() and (f.cancelled() or f.exception() is not None))
    # 无法直接区分 pending，这里置 0，交由 records 弥补
    pending = 0
    return total, running, completed, failed, pending

# ---- TaskPoolManager 内新增：对外暴露的 stats() ----
def stats(self):
    """
    返回池化运行时统计，尽量不依赖具体实现。
    需要 TaskPoolManager 上存在以下常见属性（若缺失会优雅降级）：
      - self.max_workers
      - self.use_process / self._use_process
      - self._queue
      - self._executor
      - self._records: {task_id: {"state": ... , ...}}
      - self._futures: {task_id: Future}
    """
    # 并发/执行器信息
    max_workers = getattr(self, "max_workers", None) or getattr(self, "_max_workers", None)
    use_process = getattr(self, "use_process", None)
    if use_process is None:
        use_process = getattr(self, "_use_process", None)
    executor_name = None
    if hasattr(self, "_executor") and self._executor is not None:
        try:
            executor_name = type(self._executor).__name__
        except Exception:
            executor_name = "Executor"

    # 队列长度
    queued = None
    if hasattr(self, "_queue"):
        queued = self._qsize_safe()

    # 从 records 聚合
    total_r, running_r, completed_r, failed_r, pending_r = 0,0,0,0,0
    if hasattr(self, "_records"):
        total_r, running_r, completed_r, failed_r, pending_r = self._aggregate_from_records()


# ======= 进程/线程池实际执行的函数（与 orchestrator 解耦） =======
def _exec_one_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    这里直接复用你现有的 run_pipeline（同步），
    让池化调度负责并发与生命周期。
    """
    # 注意：进程池中要确保可序列化，不要传 logger/handler 等状态对象。
    return run_pipeline(payload)
