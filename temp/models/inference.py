import joblib
import pandas as pd
import os
import json

def model_fn(model_dir):
    # Carga el modelo entrenado (ejemplo: final_stage/Model.joblib)
    model_path = os.path.join(model_dir, "Model.joblib")
    model = joblib.load(model_path)
    # Si necesitas cargar features: features.json (opcional)
    features_path = os.path.join(model_dir, "Model.json")
    if os.path.exists(features_path):
        with open(features_path, "r") as f:
            features = list(json.load(f).keys())
    else:
        features = None
    return {"model": model, "features": features}

def input_fn(input_data, content_type):
    # Recibe el JSON y lo convierte a DataFrame
    if content_type == "application/json":
        data = json.loads(input_data)
        df = pd.DataFrame(data["data"])
        return df
    raise Exception(f"Content type {content_type} not supported.")

def predict_fn(input_data, model_dict):
    # Aplica el modelo
    model = model_dict["model"]
    features = model_dict.get("features")
    if features is not None:
        missing = set(features) - set(input_data.columns)
        for feat in missing:
            input_data[feat] = 0  # O el valor que uses para imputar
        input_data = input_data[features]
    # Predice (ajusta según sea binary, multiclass, etc)
    proba = model.predict_proba(input_data)[:, 1]
    return proba

def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps({"probabilities": prediction.tolist()})
    raise Exception(f"Accept type {accept} not supported.")