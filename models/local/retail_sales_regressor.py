
import logging
from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

def _split_cols(df, params):
    target = params["target_column"]
    num = params.get("numeric_features", [])
    cat = params.get("categorical_features", [])
    return target, num, cat

def train_func(train_dataset_path, checkpoint_path, params: dict, workspace, **kwargs):
    df = pd.read_csv(train_dataset_path)
    target, num, cat = _split_cols(df, params)
    if target not in df.columns:
        raise ValueError(f"训练集缺少目标列: {target}")
    y = df[target]
    X = df[num + cat]

    pre = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])

    rf_params = {k:v for k,v in params.items() if k not in ("target_column","numeric_features","categorical_features")}
    pipe = Pipeline([("pre", pre), ("model", RandomForestRegressor(**rf_params))])

    logger.info("[Retail TRAIN] 开始训练...")
    pipe.fit(X, y)
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, checkpoint_path)
    logger.info("[Retail TRAIN] 模型保存: %s", checkpoint_path)
    return str(checkpoint_path)

def predict_func(predict_dataset_path, checkpoint_path, params: dict, workspace, **kwargs):
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"权重不存在: {ckpt}")
    pipe = joblib.load(ckpt)

    df = pd.read_csv(predict_dataset_path)
    target, num, cat = _split_cols(df, params)
    if target in df.columns:
        df = df.drop(columns=[target])
    used = [c for c in (num + cat) if c in df.columns]
    if not used:
        raise ValueError("预测集没有任何指定特征列")
    preds = pipe.predict(df[used])

    out = workspace.output_dir / f"{workspace.job_id}_retail_predictions.csv"
    pd.DataFrame({"prediction": preds}).to_csv(out, index=False)
    logger.info("[Retail PREDICT] 写入: %s", out)
    return preds
