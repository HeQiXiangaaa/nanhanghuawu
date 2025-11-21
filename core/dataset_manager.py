# core/dataset_manager.py
"""
数据集管理模块

职责划分：
- 负责管理「全局数据仓库」 datasets/ 下的数据路径
- 提供辅助方法将全局数据集复制到某个 job 的工作区中
"""
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
GLOBAL_DATASET_DIR = BASE_DIR / "datasets"


class DatasetManager:

    @staticmethod
    def get_global_dataset_path(kind, dataset_name):
        """
        kind: "train" or "predict"
        """
        p = GLOBAL_DATASET_DIR / kind / dataset_name
        if not p.exists():
            raise FileNotFoundError(f"找不到全局数据集: {p}")
        return p

    @staticmethod
    def copy_to_workspace(src_path, dest_dir, job_id):
        """
        根据指令动态创建 train 或 predict 目录
        """
        dest_path = dest_dir / src_path.name
        shutil.copy2(src_path, dest_path)
        return dest_path

