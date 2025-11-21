import joblib, inspect
from sklearn.pipeline import Pipeline

def peek(path):
    obj = joblib.load(path)
    print("=== File:", path, "===")
    print("Type:", type(obj))
    if isinstance(obj, Pipeline):
        print("Pipeline steps:", [name for name,_ in obj.steps])
    for attr in ["feature_names_in_", "n_features_in_", "mean_", "scale_", "var_", "data_min_", "data_max_"]:
        if hasattr(obj, attr):
            v = getattr(obj, attr)
            shape = getattr(v, "shape", None)
            print(f"{attr}: shape={shape}" if shape else f"{attr}: {type(v)}")
    print()

peek(r"C:\Users\ROGmb\PycharmProjects\model_intermediary\models\pretrained_weights\online_traffic_group2_1_informer_scaler.joblib")
peek(r"C:\Users\ROGmb\PycharmProjects\model_intermediary\models\pretrained_weights\retail_sales_regressor.joblib")
