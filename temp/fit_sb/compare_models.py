import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import json

def load_model(model_path):
    return joblib.load(model_path)

def load_features_json(features_path):
    with open(features_path, "r") as f:
        features_dict = json.load(f)
    if isinstance(features_dict, dict):
        features = list(features_dict.keys())
        feature_types = features_dict
    else:
        features = features_dict
        feature_types = {}
    return features, feature_types

def load_test_data(test_path):
    return pd.read_pickle(test_path)

def compare_features(features_sb, features_dp):
    print("\n=== Feature Comparison ===")
    print(f"SB ({len(features_sb)}): {features_sb}")
    print(f"DP ({len(features_dp)}): {features_dp}")
    diff_sb = set(features_sb) - set(features_dp)
    diff_dp = set(features_dp) - set(features_sb)
    if not diff_sb and not diff_dp:
        print("✅ Features are identical!")
    else:
        print(f"❌ Features in SB only: {diff_sb}")
        print(f"❌ Features in DP only: {diff_dp}")

def compare_models(model_sb, model_dp):
    print("\n=== Model Comparison ===")
    print(f"Type SB: {type(model_sb)}")
    print(f"Type DP: {type(model_dp)}")
    print(f"SB params: {getattr(model_sb, 'get_params', lambda: 'N/A')()}")
    print(f"DP params: {getattr(model_dp, 'get_params', lambda: 'N/A')()}")

def evaluate_predictions(y_true, y_pred_sb, y_pred_dp):
    print("\n=== Metrics Comparison ===")
    print("SB Model:")
    print(classification_report(y_true, y_pred_sb, digits=4))
    print("DP Model:")
    print(classification_report(y_true, y_pred_dp, digits=4))
    match = np.all(y_pred_sb == y_pred_dp)
    print(f"\nDo predictions match exactly? {'✅ YES' if match else '❌ NO'}")
    print(f"Percent matching predictions: {100*np.mean(y_pred_sb == y_pred_dp):.2f}%")

if __name__ == "__main__":
    # ------ Ajusta los nombres/rutas según los que uses ------
    model_sb_path = "temp/fit_sb/model_SB.joblib"
    model_dp_path = "temp/fit_sb/model_DP.joblib"
    features_sb_path = "temp/fit_sb/model_SB.json"
    features_dp_path = "temp/fit_sb/model_DP.json"
    test_data_path = "temp/fit_sb/X_test.pkl"
    y_test_path = "temp/fit_sb/y_test.pkl"

    # Carga modelos y features + types
    model_sb = load_model(model_sb_path)
    model_dp = load_model(model_dp_path)
    features_sb, feature_types_sb = load_features_json(features_sb_path)
    features_dp, feature_types_dp = load_features_json(features_dp_path)
    X_test = load_test_data(test_data_path)
    y_test = load_test_data(y_test_path)
    
    # Descubre automáticamente la columna de target si y_test es un DataFrame
    if isinstance(y_test, pd.DataFrame):
        if y_test.shape[1] == 1:
            y_true = y_test.iloc[:, 0]
        else:
            print(f"⚠️ y_test tiene varias columnas: {y_test.columns}")
            y_true = y_test[y_test.columns[0]]
    else:
        y_true = y_test

    # Usa los features del modelo SB, agrega si faltan
    for col in features_sb:
        if col not in X_test.columns:
            print(f"⚠️ Agregando columna faltante en X_test: {col}")
            X_test[col] = np.nan

    # Forzar dtype correcto antes del predict
    for col in features_sb:
        if feature_types_sb.get(col) == "category" or feature_types_sb.get(col) == "object":
            X_test[col] = X_test[col].astype("category")

    # Ordena columnas para el modelo SB y DP
    X_test_sb = X_test[features_sb].copy()
    X_test_dp = X_test[features_dp].copy()

    # Predicciones
    y_pred_sb = model_sb.predict(X_test_sb)
    y_pred_dp = model_dp.predict(X_test_dp)

    # Comparaciones
    compare_features(features_sb, features_dp)
    compare_models(model_sb, model_dp)
    evaluate_predictions(y_true, y_pred_sb, y_pred_dp)