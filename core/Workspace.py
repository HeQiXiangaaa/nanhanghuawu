# core/Workspace.py
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "jobs_storage"


class JobWorkspace:
    """
    目录结构改为：
      jobs_storage/
        ├─ _tmp/
        │   └─ {YYYYMMDD}/job_{job_id}/...
        └─ {YYYYMMDD}/job_{job_id}/...
    """

    def __init__(self, job_id: str, date_str: str | None = None):
        self.job_id = job_id
        self.date_str = date_str or datetime.now().strftime("%Y%m%d")

        # ✅ 最终/临时根目录都按日期分一层，不再在名字加前缀
        self.final_dir = STORAGE_DIR / self.date_str / f"job_{job_id}"
        self.temp_root = STORAGE_DIR / "_tmp" / self.date_str / f"job_{job_id}"
        self.temp_root.mkdir(parents=True, exist_ok=True)

        # 模型 & 输出目录（立即创建）
        self.models_dir = self.temp_root / "models"
        self.outputs_dir = self.temp_root / "outputs"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def train_dir(self) -> Path:
        d = self.temp_root / "datasets" / "train"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def predict_dir(self) -> Path:
        d = self.temp_root / "datasets" / "predict"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def output_dir(self) -> Path:
        return self.outputs_dir

    def finalize(self) -> Path:
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.temp_root), str(self.final_dir))
        return self.final_dir

    def cleanup(self):
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root, ignore_errors=True)
