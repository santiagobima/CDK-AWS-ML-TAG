import joblib
import json
import pandas as pd
import numpy as np

def load_model(path):
    return joblib.load(path)

def load_features_json(path):
    with open(path, "r") as f:
        features_dict = json.load(f)
    features = list(features_dict.keys())
    return features, features_dict

def build_dummy_dataframe(feature_list, feature_types, n_rows=5):
    df = pd.DataFrame(index=range(n_rows))
    for col in feature_list:
        dtype = feature_types.get(col, "float64")
        if dtype in ("category", "object"):
            df[col] = pd.Series(["unknown"] * n_rows, dtype="category")
        elif "int" in dtype.lower():
            df[col] = 0
        else:
            df[col] = 0.0
    return df

if __name__ == "__main__":
    # Paths
    model_sb_path = "temp/fit_sb/model_SB.joblib"
    model_dp_path = "temp/fit_sb/model_DP.joblib"
    features_sb_path = "temp/fit_sb/model_SB.json"
    features_dp_path = "temp/fit_sb/model_DP.json"

    # Load models and features
    model_sb = load_model(model_sb_path)
    model_dp = load_model(model_dp_path)
    features_sb, types_sb = load_features_json(features_sb_path)
    features_dp, types_dp = load_features_json(features_dp_path)

    # Build dummy input for each model
    df_sb = build_dummy_dataframe(features_sb, types_sb, n_rows=5)
    df_dp = build_dummy_dataframe(features_dp, types_dp, n_rows=5)

    print(f"🔢 Model_SB expects {len(features_sb)} features")
    print(f"🔢 Model_DP expects {len(features_dp)} features")

    # Predict
    print("\n=== Predictions on dummy data ===")
    pred_sb = model_sb.predict(df_sb)
    pred_dp = model_dp.predict(df_dp)

    for i in range(len(df_sb)):
        print(f"Row {i}: SB = {pred_sb[i]} | DP = {pred_dp[i]}")

    match_ratio = np.mean(pred_sb == pred_dp)
    print(f"\n✅ Matching predictions: {100 * match_ratio:.2f}%")