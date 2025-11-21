# core/base_wrapper.py

class BaseModelWrapper:
    """
    所有具体模型 wrapper 的父类。
    ModelManager 负责给它注入：
      - model_name
      - checkpoint_path
      - model_params
      - workspace
    """

    def __init__(self, model_name, checkpoint_path, model_params=None, workspace=None):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.model_params = model_params or {}
        self.workspace = workspace

    def train(self, train_dataset_path: str, **kwargs):
        raise NotImplementedError

    def predict(self, predict_dataset_path: str, **kwargs):
        raise NotImplementedError