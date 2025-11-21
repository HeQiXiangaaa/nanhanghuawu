# models/local/house_complex_regressor.py
# 房屋相关复杂回归模型（推测场景：房价预测/房屋相关指标回归）
# 核心功能：基于随机森林回归算法，实现数据预处理→模型训练→预测全流程
# 支持数值特征标准化、分类特征独热编码，模型训练结果持久化，预测结果本地落盘

import logging  # 日志模块：记录训练/预测过程中的关键信息（如开始训练、模型保存路径）
from pathlib import Path  # 路径处理模块：跨平台兼容的文件/目录路径操作

import joblib  # 模型序列化模块：高效保存/加载sklearn模型（比pickle更适合大数据模型）
import pandas as pd  # 数据处理模块：读取CSV数据、数据清洗与特征提取
from sklearn.compose import ColumnTransformer  # 特征处理器：对不同类型特征应用不同预处理逻辑
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归器：核心模型（适用于非线性回归场景）
from sklearn.pipeline import Pipeline  # 流水线模块：串联预处理步骤与模型，避免数据泄露
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 预处理工具：标准化、独热编码

# 初始化日志器：继承全局日志配置（参考logging_config.py），记录模块内关键操作
logger = logging.getLogger(__name__)


def _split_columns(df, params):
    """
    从数据框中拆分目标列、数值特征列、分类特征列（内部辅助函数，不对外暴露）

    参数:
        df (pd.DataFrame): 输入数据框（训练集/预测集）
        params (dict): 配置字典，需包含以下键：
            - target_column: 目标列名称（如房价预测中的"house_price"）
            - numeric_features: 数值特征列名列表（如"square_footage"、"room_count"）
            - categorical_features: 分类特征列名列表（如"district"、"house_type"）

    返回:
        tuple: (target_col, num_cols, cat_cols)
            - target_col (str): 目标列名称
            - num_cols (list): 数值特征列名列表（可能为空）
            - cat_cols (list): 分类特征列名列表（可能为空）
    """
    target_col = params["target_column"]
    # 从配置中获取数值/分类特征，若未配置则默认空列表（避免KeyError）
    num_cols = params.get("numeric_features", [])
    cat_cols = params.get("categorical_features", [])
    return target_col, num_cols, cat_cols


def train_func(train_dataset_path, checkpoint_path, params: dict, workspace, **kwargs):
    """
    模型训练主函数：读取训练数据→数据预处理→模型训练→保存模型

    参数:
        train_dataset_path (str): 训练集CSV文件路径（本地文件系统路径）
        checkpoint_path (str): 模型权重保存路径（最终保存为.joblib格式）
        params (dict): 训练配置字典，包含：
            - 必需键：target_column、numeric_features、categorical_features（特征相关）
            - 可选键：n_estimators、max_depth等（RandomForestRegressor的超参数）
        workspace: 工作空间对象（参考Workspace.py），用于管理任务目录、输出路径等
        **kwargs: 额外扩展参数（预留接口，如自定义回调函数）

    返回:
        str: 模型权重保存路径（与输入checkpoint_path一致，便于后续调用）

    异常:
        ValueError: 训练集缺少配置中指定的目标列时抛出
    """
    # 1. 读取训练数据（CSV格式）
    df = pd.read_csv(train_dataset_path)

    # 2. 拆分目标列、数值特征列、分类特征列
    target_col, num_cols, cat_cols = _split_columns(df, params)

    # 校验目标列是否存在（避免因配置错误导致训练失败）
    if target_col not in df.columns:
        raise ValueError(f"训练集缺少目标列: {target_col}（请检查params中的target_column配置）")

    # 3. 分离特征矩阵X和目标变量y
    y = df[target_col]  # 目标变量（如房价）
    X = df[num_cols + cat_cols]  # 特征矩阵（数值特征+分类特征拼接）

    # 4. 构建数据预处理流水线（按特征类型分别处理，避免数据泄露）
    preprocessor = ColumnTransformer(
        transformers=[
            # 数值特征处理：标准化（StandardScaler）→ 均值为0，方差为1，提升模型收敛速度
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            # 分类特征处理：独热编码（OneHotEncoder）→ 将离散类别转为二进制向量
            # handle_unknown='ignore'：忽略未见过的类别（避免预测时因新类别报错）
            ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
        ],
        remainder="drop"  # 未指定的特征列直接丢弃（避免无关特征干扰）
    )

    # 5. 提取随机森林超参数（过滤掉特征相关配置，仅保留模型参数）
    rf_model_params = {
        k: v for k, v in params.items()
        if k not in ("target_column", "numeric_features", "categorical_features")
    }

    # 6. 构建完整训练流水线：预处理 → 模型训练（串联执行，确保预处理仅作用于训练集）
    training_pipeline = Pipeline([
        ("preprocessor", preprocessor),  # 第一步：数据预处理
        ("model", RandomForestRegressor(**rf_model_params))  # 第二步：随机森林回归训练
    ])

    # 7. 启动模型训练
    logger.info(f"[CFG TRAIN] 开始训练随机森林回归模型...")
    logger.info(f"[CFG TRAIN] 模型超参数: {rf_model_params}")
    logger.info(f"[CFG TRAIN] 目标列: {target_col}, 数值特征数: {len(num_cols)}, 分类特征数: {len(cat_cols)}")
    training_pipeline.fit(X, y)  # 拟合数据（预处理和训练一体化执行）

    # 8. 保存模型（确保保存目录存在，避免FileNotFoundError）
    checkpoint_dir = Path(checkpoint_path).parent  # 获取模型保存目录路径
    checkpoint_dir.mkdir(parents=True, exist_ok=True)  # 递归创建目录（父母目录不存在则创建，已存在则忽略）
    joblib.dump(training_pipeline, checkpoint_path)  # 序列化保存整个流水线（含预处理逻辑+模型权重）

    logger.info(f"[CFG TRAIN] 模型训练完成，已保存至: {checkpoint_path}")
    return str(checkpoint_path)  # 返回模型路径，便于后续预测调用


def predict_func(predict_dataset_path, checkpoint_path, params: dict, workspace, **kwargs):
    """
    模型预测主函数：加载训练好的模型→读取预测数据→预处理→预测→保存结果

    参数:
        predict_dataset_path (str): 预测集CSV文件路径（本地文件系统路径）
        checkpoint_path (str): 模型权重路径（train_func保存的.joblib文件）
        params (dict): 预测配置字典（与训练时一致，需包含target_column、numeric_features、categorical_features）
        workspace: 工作空间对象（参考Workspace.py），用于获取输出目录、任务ID等
        **kwargs: 额外扩展参数（预留接口，如预测批次大小控制）

    返回:
        np.ndarray: 预测结果数组（与预测集行数一致）

    异常:
        FileNotFoundError: 模型权重文件不存在时抛出
    """
    # 1. 校验模型权重文件是否存在
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"找不到模型权重文件: {checkpoint_file.absolute()}（请先执行训练函数）")

    # 2. 加载训练好的完整流水线（含预处理逻辑，确保预测时预处理方式与训练一致）
    trained_pipeline = joblib.load(checkpoint_file)
    logger.info(f"[CFG PREDICT] 已加载模型权重: {checkpoint_path}")

    # 3. 读取预测数据（CSV格式）
    df_predict = pd.read_csv(predict_dataset_path)

    # 4. 拆分特征列（使用与训练一致的配置，确保特征匹配）
    target_col, num_cols, cat_cols = _split_columns(df_predict, params)

    # 5. 处理目标列（若预测集中包含目标列，直接删除，避免数据泄露）
    if target_col in df_predict.columns:
        df_predict = df_predict.drop(columns=[target_col])
        logger.info(f"[CFG PREDICT] 预测集包含目标列{target_col}，已自动删除")

    # 6. 筛选可用特征列（避免预测集缺少部分特征导致报错）
    # 仅保留"配置中指定的特征"且"预测集中存在的列"
    available_features = [c for c in (num_cols + cat_cols) if c in df_predict.columns]
    missing_features = set(num_cols + cat_cols) - set(available_features)
    if missing_features:
        logger.warning(f"[CFG PREDICT] 预测集缺少部分配置特征: {missing_features}（已自动忽略）")

    # 构建预测用特征矩阵
    X_predict = df_predict[available_features]

    # 7. 执行预测（流水线自动应用训练时的预处理逻辑，无需手动重复）
    logger.info(f"[CFG PREDICT] 开始预测... 预测样本数: {len(X_predict)}, 使用特征数: {len(available_features)}")
    predictions = trained_pipeline.predict(X_predict)

    # 8. 保存预测结果到本地（按工作空间规范存储）
    output_dir = workspace.output_dir  # 从工作空间获取输出目录（统一管理任务输出）
    output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    # 构建输出文件名：任务ID_预测集文件名_pred.csv（便于追溯任务和数据源）
    predict_filename = Path(predict_dataset_path).stem  # 获取预测集文件名（不含后缀）
    output_file_path = output_dir / f"{workspace.job_id}_{predict_filename}_pred.csv"

    # 保存预测结果为CSV（仅含预测列，便于后续分析）
    pd.DataFrame({"prediction": predictions}).to_csv(output_file_path, index=False)
    logger.info(f"[CFG PREDICT] 预测结果已写入: {output_file_path}")

    return predictions  # 返回预测结果数组（支持后续二次处理，如集成其他系统）

