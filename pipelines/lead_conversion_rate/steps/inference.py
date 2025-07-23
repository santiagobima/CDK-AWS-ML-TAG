import joblib
import numpy as np
import json
import os
import subprocess
import sys
import pandas as pd


def model_fn(model_dir):
    # No cargamos modelos aquí, lo hacemos en predict_fn dinámicamente
    return {"model_dir": model_dir}

def input_fn(request_body, content_type):
    if content_type == 'application/json':
        data = json.loads(request_body)
        stage = data.get('stage')
        if stage is None:
            raise ValueError("Missing 'stage' in request JSON.")
        df = pd.DataFrame(data['data'])
        return {"stage": stage, "data": df}
    raise Exception(f'Content type {content_type} not supported')

def predict_fn(input_dict, model_dict):
    stage = input_dict["stage"]
    input_data = input_dict["data"]
    model_dir = model_dict["model_dir"]

    # Ruta específica del stage
    stage_dir = os.path.join(model_dir, stage)
    model_path = os.path.join(stage_dir, "Model.joblib")
    features_path = os.path.join(stage_dir, "Model.json")

    if not os.path.exists(model_path) or not os.path.exists(features_path):
        raise FileNotFoundError(f"Modelo o features no encontrados para el stage '{stage}'")

    # Cargar modelo y features
    model = joblib.load(model_path)
    with open(features_path, "r") as f:
        features_types = json.load(f)

    # Preprocesamiento: tipado + columnas faltantes
    for col, dtype in features_types.items():
        if col in input_data.columns:
            try:
                input_data[col] = input_data[col].astype(dtype)
            except Exception:
                pass  # Silencioso si falla
        else:
            input_data[col] = 0

    input_data = input_data[list(features_types.keys())]

    # Predicción
    proba = model.predict_proba(input_data)[:, 1]
    return proba

def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps({"probabilities": prediction.tolist()})
    raise Exception(f"Accept type {accept} not supported.")