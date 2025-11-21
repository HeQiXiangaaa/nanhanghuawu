# core/model_manager.py

import importlib  # 用于动态导入模块
import importlib.util
import json
import logging
import shutil  # 用于文件复制操作
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict

from core.base_wrapper import BaseModelWrapper  # 模型包装器基类

logger = logging.getLogger(__name__)

# 路径定义
BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_MODEL_DIR = BASE_DIR / "models" / "local"  # 本地模型配置文件目录
PRETRAINED_DIR = BASE_DIR / "models" / "pretrained_weights"  # 预训练权重目录
ALLOWED_WEIGHT_EXTS = [".ckpt", ".joblib", ".pkl", "pt"]  # 支持的权重文件扩展名


# ---------- 适配器：将普通函数包装为 BaseModelWrapper 兼容接口 ----------
class FunctionWrapper(BaseModelWrapper):
    """
    适配“配置文件+自定义函数”形式的模型包装器
    功能：将动态解析的 train_func 和 predict_func 适配为 BaseModelWrapper 接口
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: Path,
        model_params: dict,
        workspace,
        train_func: Optional[Callable] = None,
        predict_func: Optional[Callable] = None,
    ):
        super().__init__(model_name, checkpoint_path, model_params, workspace)
        self._train_func = train_func  # 动态加载的训练函数
        self._predict_func = predict_func  # 动态加载的预测函数

    def train(self, train_dataset_path: str, **kwargs):
        if self._train_func is None:
            raise RuntimeError(f"模型 {self.model_name} 未提供训练函数")
        # 调用格式：train_func(数据集路径, 检查点路径, 参数, 工作区, 其他参数)
        return self._train_func(
            train_dataset_path=train_dataset_path,
            checkpoint_path=self.checkpoint_path,
            params=self.model_params,
            workspace=self.workspace,** kwargs,
        )

    def predict(self, predict_dataset_path: str, **kwargs):
        if self._predict_func is None:
            raise RuntimeError(f"模型 {self.model_name} 未提供预测函数")
        # 调用格式：predict_func(数据集路径, 检查点路径, 参数, 工作区, 其他参数)
        return self._predict_func(
            predict_dataset_path=predict_dataset_path,
            checkpoint_path=self.checkpoint_path,
            params=self.model_params,
            workspace=self.workspace,** kwargs,
        )


# ---------- 从配置文件解析模型函数 ----------
def _load_funcs_from_config(model_name: str) -> Tuple[Callable, Callable, Dict]:
    """
    从本地模型配置文件加载训练/预测函数及默认参数
    配置文件路径：models/local/<model_name>_config.json
    配置结构：
      - train_function: "module.func" 形式的训练函数路径
      - predict_function: "module.func" 形式的预测函数路径
      - params: 模型默认超参字典
    返回：(训练函数, 预测函数, 默认参数)
    """
    config_path = LOCAL_MODEL_DIR / f"{model_name}_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"模型配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)
    def _resolve(call_path: str) -> Callable:
        """解析"module.func"字符串为实际可调用函数"""
        if "." not in call_path:
            raise ValueError(f"函数路径格式错误: {call_path}（需为'module.func'）")
        module_name, func_name = call_path.rsplit(".", 1)
        full_module = f"models.local.{module_name}"  # 补全模块路径
        module = importlib.import_module(full_module)
        func = getattr(module, func_name, None)
        if func is None or not callable(func):
            raise AttributeError(f"{full_module} 中未找到可调用函数 {func_name}")
        return func

    train_func = _resolve(cfg["train_function"])
    predict_func = _resolve(cfg["predict_function"])
    default_params = cfg.get("params", {}) or {}
    return train_func, predict_func, default_params


def _find_pretrained_weight(model_name: str) -> Optional[Path]:
    """
    在预训练权重目录查找模型对应的权重文件
    优先检查 ALLOWED_WEIGHT_EXTS 扩展名，未找到则返回任意匹配文件
    返回：找到的权重路径或None
    """
    for ext in ALLOWED_WEIGHT_EXTS:
        weight_path = PRETRAINED_DIR / f"{model_name}{ext}"
        if weight_path.exists():
            return weight_path
    # 兜底：查找任意以模型名开头的文件
    candidates = list(PRETRAINED_DIR.glob(f"{model_name}.*"))
    return candidates[0] if candidates else None


class ModelManager:
    """模型管理器：负责加载模型包装器，处理配置解析和权重管理"""

    @staticmethod
    def get_model_wrapper(task_type: str, model_name: str, model_params: dict, workspace) -> BaseModelWrapper:
        """
        获取模型包装器（核心方法）
        流程：
          1. 解析模型配置文件，加载训练/预测函数及默认参数
          2. 合并默认参数与请求参数（请求参数优先）
          3. 定义任务专属检查点路径
          4. 预测任务：若无任务检查点，自动回退到全局预训练权重
          5. 返回适配后的 FunctionWrapper 实例
        """
        # 1. 解析配置并合并参数
        train_func, predict_func, default_params = _load_funcs_from_config(model_name)
        merged_params = dict(default_params)
        merged_params.update(model_params or {})  # 请求参数覆盖默认参数

        # 2. 定义任务内的模型检查点路径
        job_ckpt = workspace.models_dir / f"{model_name}.ckpt"
        job_ckpt.parent.mkdir(parents=True, exist_ok=True)

        # 3. 预测任务权重回退逻辑：无任务检查点则复制预训练权重
        if task_type == "predict" and not job_ckpt.exists():
            pretrained_weight = _find_pretrained_weight(model_name)
            if pretrained_weight is not None:
                shutil.copy2(pretrained_weight, job_ckpt)
                logger.info("[MODEL] 预测权重回退：%s -> %s", pretrained_weight, job_ckpt)
            else:
                raise FileNotFoundError(
                    f"预测任务无可用权重：\n"
                    f"- 任务检查点: {job_ckpt}\n- 预训练目录: {PRETRAINED_DIR}"
                )

        logger.info(
            "[MODEL] 任务=%s 模型=%s 类型=%s 检查点=%s",
            workspace.job_id, model_name, task_type, job_ckpt
        )

        # 4. 返回包装器实例
        return FunctionWrapper(
            model_name=model_name,
            checkpoint_path=job_ckpt,
            model_params=merged_params,
            workspace=workspace,
            train_func=train_func,
            predict_func=predict_func,
        )