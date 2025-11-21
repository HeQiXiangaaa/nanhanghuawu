# core/orchestrator.py
import uuid
import logging
import re, json, time

from typing import Dict, Any

from core.Workspace import JobWorkspace  # 任务工作空间管理（文件、路径等）
from core.dataset_manager import DatasetManager  # 数据集管理（获取、复制等）
from core.model_manager import ModelManager  # 模型管理（获取模型包装器）

logger = logging.getLogger(__name__)

def run_pipeline(request_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    任务编排入口，处理三种任务类型：
      - train: 仅训练模型
      - predict: 仅使用模型预测
      - train_and_predict: 先训练再预测
    参数: request_dict 包含任务配置的字典
    返回: 包含任务结果的字典
    """

    # 提取请求参数
    task_type = request_dict.get("task_type")
    dataset_name = request_dict.get("dataset_name")
    predict_dataset_name = request_dict.get("predict_dataset_name")
    model_name = request_dict.get("model_name")
    model_params = request_dict.get("model_params") or {}

    # 验证必要参数合法性
    if task_type not in {"train", "predict", "train_and_predict"}:
        raise ValueError(f"非法 task_type: {task_type}")
    if not dataset_name:
        raise ValueError("dataset_name 不能为空")
    if not model_name:
        raise ValueError("model_name 不能为空")

    # 生成任务ID并初始化工作空间（用于存储任务相关文件）
    job_id = request_dict.get("force_job_id")
    workspace = JobWorkspace(job_id)

    # 为当前任务添加独立日志处理器（记录任务专属日志）
    root_logger = logging.getLogger()
    job_log_path = workspace.output_dir / "job.log"
    job_handler = logging.FileHandler(job_log_path, encoding="utf-8")
    job_handler.setLevel(logging.INFO)
    job_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    ))
    root_logger.addHandler(job_handler)

    logger.info(
        "[JOB %s] 新任务: task_type=%s, dataset_name=%s, predict_dataset_name=%s, "
        "model_name=%s, model_params=%s",
        job_id, task_type, dataset_name, predict_dataset_name, model_name, model_params,
    )

    # 初始化返回结果字典
    result: Dict[str, Any] = {
        "job_id": job_id,
        "task_type": task_type,
        "model_name": model_name,
        "model_params": model_params,
    }

    try:
        # 获取模型包装器（封装了模型的训练/预测逻辑）
        wrapper = ModelManager.get_model_wrapper(
            task_type=task_type,
            model_name=model_name,
            model_params=model_params,
            workspace=workspace,
        )

        # 处理仅训练任务
        if task_type == "train":
            # 获取训练数据集并复制到工作空间
            src = DatasetManager.get_global_dataset_path("train", dataset_name)
            train_path = DatasetManager.copy_to_workspace(src, workspace.train_dir, job_id)

            # 调用模型训练接口
            ckpt_path = wrapper.train(train_dataset_path=str(train_path), job_id=job_id)

            # 清理日志处理器，固化工作空间并更新结果
            root_logger.removeHandler(job_handler)
            job_handler.close()
            final_dir = workspace.finalize()

            result.update({
                "status": "train_finished",
                "workspace": str(final_dir),
                "train_dataset": str(train_path),
                "model_checkpoint_path": str(ckpt_path),
                "job_log_path": str(final_dir / "outputs" / "job.log"),
            })
            logger.info("[JOB %s] 训练任务完成", job_id)
            return result

        # 处理仅预测任务
        if task_type == "predict":
            # 获取预测数据集并复制到工作空间
            src = DatasetManager.get_global_dataset_path("predict", dataset_name)
            predict_path = DatasetManager.copy_to_workspace(src, workspace.predict_dir, job_id)

            # 调用模型预测接口，处理预测结果格式
            preds = wrapper.predict(predict_dataset_path=str(predict_path), job_id=job_id)
            if hasattr(preds, "tolist"):
                preds = preds.tolist()

            # 清理资源并更新结果
            root_logger.removeHandler(job_handler)
            job_handler.close()
            final_dir = workspace.finalize()

            result.update({
                "status": "predict_finished",
                "workspace": str(final_dir),
                "predict_dataset": str(predict_path),
                "predictions_preview": preds[:10],  # 返回前10条预测结果预览
                "job_log_path": str(final_dir / "outputs" / "job.log"),
            })
            logger.info("[JOB %s] 预测任务完成", job_id)
            return result

        # 处理训练+预测任务
        if task_type == "train_and_predict":
            # 训练阶段：获取并复制训练数据集，执行训练
            src_train = DatasetManager.get_global_dataset_path("train", dataset_name)
            train_path = DatasetManager.copy_to_workspace(src_train, workspace.train_dir, job_id)
            ckpt_path = wrapper.train(train_dataset_path=str(train_path), job_id=job_id)

            # 预测阶段：获取并复制预测数据集（默认使用训练数据集），执行预测
            pred_name = predict_dataset_name or dataset_name
            src_pred = DatasetManager.get_global_dataset_path("predict", pred_name)
            predict_path = DatasetManager.copy_to_workspace(src_pred, workspace.predict_dir, job_id)
            preds = wrapper.predict(predict_dataset_path=str(predict_path), job_id=job_id)
            if hasattr(preds, "tolist"):
                preds = preds.tolist()

            # 清理资源并更新结果
            root_logger.removeHandler(job_handler)
            job_handler.close()
            final_dir = workspace.finalize()

            result.update({
                "status": "train_and_predict_finished",
                "workspace": str(final_dir),
                "train_dataset": str(train_path),
                "predict_dataset": str(predict_path),
                "model_checkpoint_path": str(ckpt_path),
                "predictions_preview": preds[:10],
                "job_log_path": str(final_dir / "outputs" / "job.log"),
            })
            logger.info("[JOB %s] 训练+预测任务完成", job_id)
            return result

        # 理论上不会触发的异常（参数验证已覆盖所有类型）
        raise ValueError(f"未知 task_type: {task_type}")

    except Exception:
        # 任务失败：记录异常日志，清理工作空间并抛出异常
        logger.exception("[JOB %s] 任务失败", job_id)
        root_logger.removeHandler(job_handler)
        job_handler.close()
        workspace.cleanup()
        raise




