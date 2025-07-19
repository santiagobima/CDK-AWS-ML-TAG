import joblib
import pandas as pd
import numpy as np
import json
import os

def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'Model.joblib')
    model = joblib.load(model_path)
    features_path = os.path.join(model_dir, 'Model.json')
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features_types = json.load(f)
    else:
        features_types = None
    return {'model': model, 'features_types': features_types}

def input_fn(request_body, content_type):
    if content_type == 'application/json':
        data = json.loads(request_body)
        df = pd.DataFrame(data['data'])
        return df
    raise Exception(f'Content type {content_type} not supported')

def predict_fn(input_data, model_dict):
    model = model_dict['model']
    features_types = model_dict.get('features_types')
    # Forzamos tipos solo si hay info de features
    if features_types is not None:
        for col, dtype in features_types.items():
            if col in input_data.columns:
                if dtype == 'category':
                    input_data[col] = input_data[col].astype('category')
                else:
                    try:
                        input_data[col] = input_data[col].astype(dtype)
                    except Exception:
                        pass  # No forzar si falla
            else:
                # Si falta, agrega con valor por defecto
                input_data[col] = 0
        # Ordenamos columnas como en el entrenamiento
        input_data = input_data[list(features_types.keys())]
    # Inferencia
    proba = model.predict_proba(input_data)[:, 1]
    return proba

def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps({"probabilities": prediction.tolist()})
    raise Exception(f"Accept type {accept} not supported.")


