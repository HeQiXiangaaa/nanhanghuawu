# models/local/sk_rf_regressor.py
# import logging
# from pathlib import Path
#
# import joblib
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
#
# logger = logging.getLogger(__name__)
#
# def _split_params(params: dict):
#     # 统一取参：目标列 + 其余超参给 RF
#     target_col = params.get("target_column", "target")
#     rf_params = dict(params or {})
#     rf_params.pop("target_column", None)
#     return target_col, rf_params
#
# def train_func(train_dataset_path, checkpoint_path, params: dict, workspace, **kwargs):
#     """
#     训练：读取 CSV -> get_dummies -> 训练 RandomForest -> 保存 {model, columns}
#     """
#     df = pd.read_csv(train_dataset_path)
#     target_col, rf_params = _split_params(params)
#
#     if target_col not in df.columns:
#         raise ValueError(f"训练集缺少目标列: {target_col}")
#
#     y = df[target_col]
#     X = pd.get_dummies(df.drop(columns=[target_col]))
#     train_columns = list(X.columns)
#
#     model = RandomForestRegressor(**rf_params)
#     logger.info("[RF TRAIN] params=%s, X=%s, y=%s", rf_params, X.shape, y.shape)
#     model.fit(X, y)
#
#     ckpt = Path(checkpoint_path)
#     ckpt.parent.mkdir(parents=True, exist_ok=True)
#
#     # 保存模型与训练时的列集合，便于预测对齐
#     payload = {"model": model, "columns": train_columns}
#     joblib.dump(payload, ckpt)
#     logger.info("[RF TRAIN] 模型已保存: %s", ckpt)
#     return str(ckpt)
#
# def predict_func(predict_dataset_path, checkpoint_path, params: dict, workspace, **kwargs):
#     """
#     预测：读取 CSV -> 丢弃 target(如有) -> get_dummies -> 按训练列集合 reindex -> predict
#          输出写到 workspace.output_dir / "<job_id>_<stem>_pred.csv"
#     """
#     ckpt = Path(checkpoint_path)
#     if not ckpt.exists():
#         raise FileNotFoundError(f"[RF PREDICT] 找不到模型权重: {ckpt}")
#
#     payload = joblib.load(ckpt)
#     model = payload["model"]
#     train_columns = payload["columns"]
#
#     df = pd.read_csv(predict_dataset_path)
#     target_col, _ = _split_params(params)
#     if target_col in df.columns:
#         df = df.drop(columns=[target_col])
#
#     X = pd.get_dummies(df)
#     # 关键：按训练时的列对齐，缺失补 0，多余列丢弃
#     X = X.reindex(columns=train_columns, fill_value=0)
#
#     preds = model.predict(X)
#
#     out_dir = workspace.output_dir
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_path = out_dir / f"{workspace.job_id}_{Path(predict_dataset_path).stem}_pred.csv"
#     pd.DataFrame({"prediction": preds}).to_csv(out_path, index=False)
#     logger.info("[RF PREDICT] 预测结果写入: %s", out_path)
#
#     return preds
#
#

# models/local/sk_rf_regressor.py
import json
import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

META_SUFFIX = ".cols.json"   # 侧车文件：保存训练后的列名列表

def _ohe(df: pd.DataFrame, target: str | None):
    if target and target in df.columns:
        y = df[target]
        X = pd.get_dummies(df.drop(columns=[target]))
        return X, y
    X = pd.get_dummies(df)
    return X, None

def _save_cols(meta_path: Path, columns: list[str]):
    meta_path.write_text(json.dumps({"columns": columns}, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_cols(meta_path: Path) -> list[str] | None:
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text("utf-8")).get("columns")

def train_func(train_dataset_path, checkpoint_path, params: dict, workspace, **kwargs):
    df = pd.read_csv(train_dataset_path)
    target = params.get("target_column", "target")
    if target not in df.columns:
        raise ValueError(f"[RF TRAIN] 训练集缺少目标列: {target}")

    X, y = _ohe(df, target)
    rf_params = dict(params); rf_params.pop("target_column", None)
    model = RandomForestRegressor(**rf_params)

    logger.info("[RF TRAIN] params=%s, X=%s, y=%s", rf_params, X.shape, y.shape)
    model.fit(X, y)

    ckpt = Path(checkpoint_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ckpt)

    # 同步写出列清单
    _save_cols(ckpt.with_suffix(ckpt.suffix + META_SUFFIX), list(X.columns))
    logger.info("[RF TRAIN] 模型与列清单已保存: %s / %s", ckpt, ckpt.with_suffix(ckpt.suffix + META_SUFFIX))
    return str(ckpt)

def predict_func(predict_dataset_path, checkpoint_path, params: dict, workspace, **kwargs):
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"[RF PREDICT] 找不到模型权重: {ckpt}")

    model = joblib.load(ckpt)
    df = pd.read_csv(predict_dataset_path)
    target = params.get("target_column", "target")
    X, _ = _ohe(df, None if target not in df.columns else target)

    # 读取训练期的列清单并对齐
    meta_cols = _load_cols(ckpt.with_suffix(ckpt.suffix + META_SUFFIX))
    if meta_cols:
        missing = [c for c in meta_cols if c not in X.columns]
        extra   = [c for c in X.columns if c not in meta_cols]
        if missing or extra:
            # 友好提示（但我们仍然支持自动对齐）
            logger.warning("[RF PREDICT] 特征列不一致: 缺失=%s, 额外=%s", missing[:5], extra[:5])
        X = X.reindex(columns=meta_cols, fill_value=0)

    preds = model.predict(X)

    out_dir = workspace.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{workspace.job_id}_{Path(predict_dataset_path).stem}_pred.csv"
    pd.DataFrame({"prediction": preds}).to_csv(out_path, index=False)
    logger.info("[RF PREDICT] 预测结果写入: %s", out_path)
    return preds



