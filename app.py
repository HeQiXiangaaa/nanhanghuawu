import logging

from flask import Flask, request, jsonify
from multiprocessing import current_process
from core.logging_config import setup_logging
from core.orchestrator import run_pipeline
from core.TaskPoolManager import TaskPoolManager
from core.pool import get_pool
from pathlib import Path
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    logger.info("收到 /health 健康检查请求")
    return jsonify({"status": "ok"}), 200


BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_LOCAL_DIR = BASE_DIR / "models" / "local"
MODELS_PRE_DIR = BASE_DIR / "models" / "pretrained_weights"
for p in [DATASETS_DIR, MODELS_LOCAL_DIR, MODELS_PRE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        saved = []
        def save_one(field, target_dir: Path):
            f = request.files.get(field)
            if f and f.filename:
                name = secure_filename(f.filename)
                (target_dir / name).write_bytes(f.read())
                saved.append(str(target_dir.relative_to(BASE_DIR) / name))

        save_one("model_code", MODELS_LOCAL_DIR)
        save_one("pretrained_weight", MODELS_PRE_DIR)
        save_one("model_config", MODELS_LOCAL_DIR)

        (DATASETS_DIR / "train").mkdir(parents=True, exist_ok=True)
        (DATASETS_DIR / "predict").mkdir(parents=True, exist_ok=True)
        save_one("train_dataset", DATASETS_DIR / "train")
        save_one("predict_dataset", DATASETS_DIR / "predict")

        return jsonify({"code": 0, "msg": "上传完成", "saved": saved}), 200
    except Exception as e:
        return jsonify({"code": 500, "msg": f"upload error: {repr(e)}"}), 500

@app.route("/submit", methods=["POST"])
def submit_task():
    try:
        POOL = get_pool()
        data = request.get_json(force=True) or {}
        priority = int(data.pop("priority", 0))
        timeout = data.pop("timeout", None)
        retries = int(data.pop("max_retries", 0))
        task_id = POOL.submit(data, priority=priority, timeout=timeout, max_retries=retries)
        return jsonify({"code": 0, "msg": "queued", "task_id": task_id}), 200
    except Exception as e:
        logger.exception("submit error: %s", e)
        return jsonify({"code": 500, "msg": f"submit error: {repr(e)}"}), 500

@app.route("/status/<task_id>", methods=["GET"])
def task_status(task_id):
    try:
        POOL = get_pool()
        st = POOL.status(task_id)
        return jsonify({"code": 0, "data": st}), 200
    except KeyError:
        return jsonify({"code": 404, "msg": "task not found"}), 404
    except Exception as e:
        return jsonify({"code": 500, "msg": repr(e)}), 500

@app.route("/result/<task_id>", methods=["GET"])
def task_result(task_id):
    try:
        POOL = get_pool()
        r = POOL.result(task_id)
        return jsonify({"code": 0, "data": r}), 200
    except KeyError:
        return jsonify({"code": 404, "msg": "task not found"}), 404
    except Exception as e:
        return jsonify({"code": 500, "msg": repr(e)}), 500

@app.route("/cancel/<task_id>", methods=["POST"])
def task_cancel(task_id):
    try:
        POOL = get_pool()
        ok = POOL.cancel(task_id)
        return jsonify({"code": 0, "msg": "cancelled" if ok else "cannot cancel", "ok": ok}), 200
    except Exception as e:
        return jsonify({"code": 500, "msg": repr(e)}), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    POOL = get_pool()
    return jsonify({"code": 0, "data": POOL.metrics()}), 200

# --- 批量提交 ---
@app.route("/submit_bulk", methods=["POST"])
def submit_bulk():
    """
    Body:
    {
      "max_concurrency": 3,
      "default_timeout": 3600,
      "default_retries": 1,
      "jobs": [ {...}, {...} ]
    }
    """
    try:
        POOL = get_pool()
        data = request.get_json(force=True) or {}
        jobs = data.get("jobs", [])
        if not isinstance(jobs, list) or not jobs:
            return jsonify({"code": 400, "msg": "jobs 不能为空且必须是数组"}), 400

        # 可选：若希望前端传来的 max_concurrency 生效，这里仅做提示，不动态变更正在运行的进程池
        want_conc = int(data.get("max_concurrency", 0) or 0)
        if want_conc and want_conc != POOL.max_workers:
            # 仅提示。要真正变更，需要重建池（生产中建议通过环境变量重启服务）
            logging.warning(f"客户端请求并发={want_conc}，当前池并发={POOL.max_workers}。若需生效，请以环境变量重启服务。")

        default_timeout = data.get("default_timeout", None)
        default_retries = int(data.get("default_retries", 0))

        results = POOL.submit_many(
            jobs,
            default_timeout=default_timeout,
            default_retries=default_retries
        )
        return jsonify({
            "code": 0,
            "msg": "queued",
            "data": [{"task_id": tid, "job": job} for tid, job in results]
        }), 200
    except Exception as e:
        logging.exception("submit_bulk error: %s", e)
        return jsonify({"code": 500, "msg": f"submit_bulk error: {repr(e)}"}), 500

# --- 批量状态查询 ---
@app.route("/status_many", methods=["POST"])
def status_many():
    """
    Body: { "task_ids": ["id1","id2",...] }
    """
    try:
        POOL = get_pool()
        data = request.get_json(force=True) or {}
        tids = data.get("task_ids", [])
        if not isinstance(tids, list) or not tids:
            return jsonify({"code": 400, "msg": "task_ids 不能为空且必须是数组"}), 400
        return jsonify({"code": 0, "data": POOL.status_many(tids)}), 200
    except Exception as e:
        return jsonify({"code": 500, "msg": f"status_many error: {repr(e)}"}), 500

# --- 批量取消 ---
@app.route("/cancel_many", methods=["POST"])
def cancel_many():
    """
    Body: { "task_ids": ["id1","id2",...] }
    """
    try:
        POOL = get_pool()
        data = request.get_json(force=True) or {}
        tids = data.get("task_ids", [])
        if not isinstance(tids, list) or not tids:
            return jsonify({"code": 400, "msg": "task_ids 不能为空且必须是数组"}), 400
        return jsonify({"code": 0, "data": POOL.cancel_many(tids)}), 200
    except Exception as e:
        return jsonify({"code": 500, "msg": f"cancel_many error: {repr(e)}"}), 500

@app.route("/pool_stats", methods=["GET"])
def pool_stats():
    try:
        POOL = get_pool()
        # 兼容没有 stats() 的旧版本
        if hasattr(POOL, "stats"):
            data = POOL.stats()
        else:
            # 兜底（极简）
            data = {
                "max_workers": getattr(POOL, "max_workers", None),
                "executor": type(getattr(POOL, "_executor", object())).__name__ if hasattr(POOL, "_executor") else None
            }
        return jsonify({"code": 0, "data": data}), 200
    except Exception as e:
        import traceback
        return jsonify({"code": 500, "msg": f"stats error: {repr(e)}", "trace": traceback.format_exc()}), 500


@app.route("/run", methods=["POST"])
def run():
    """
    统一任务入口：
    - train
    - predict
    - train_and_predict
    """
    try:
        data = request.get_json(force=True) or {}
        logger.info("收到 /run 请求: %s", data)
    except Exception as e:
        logger.exception("JSON 解析失败: %s", e)
        return jsonify({"code": 400, "msg": f"JSON 解析失败: {e}"}), 400

    try:
        result = run_pipeline(data)
        logger.info(
            "任务执行完成: job_id=%s, status=%s",
            result.get("job_id"),
            result.get("status"),
        )
        return jsonify({
            "code": 0,
            "msg": "success",
            "data": result
        }), 200
    except ValueError as e:
        logger.exception("请求参数错误: %s", e)
        return jsonify({
            "code": 400,
            "msg": f"参数错误: {repr(e)}"
        }), 400
    except Exception as e:
        logger.exception("内部错误: %s", e)
        return jsonify({
            "code": 500,
            "msg": f"内部错误: {repr(e)}"
        }), 500


if __name__ == "__main__":
    # 开发环境可以保留 debug=True，生产环境建议改为 False
    app.run(host="0.0.0.0", port=9655, debug=False, use_reloader=False)
