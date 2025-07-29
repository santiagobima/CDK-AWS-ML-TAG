import os
import tarfile
import joblib
import json
import pandas as pd

# Ruta al tar.gz en local
local_tar_path = "temp/lambda/multiendpoint.tar.gz"
extraction_dir = "/tmp/multi_model_test"
stage = "init_stage"
stage_path = os.path.join(extraction_dir, stage)

# Crear carpeta destino si no existe
os.makedirs(extraction_dir, exist_ok=True)

# Extraer tar.gz
if not os.path.exists(stage_path):
    print(f"📦 Extrayendo {local_tar_path} en {extraction_dir}...")
    with tarfile.open(local_tar_path, "r:gz") as tar:
        tar.extractall(path=extraction_dir)
    print("✅ Extracción completada.")

# Cargar modelo
model_path = os.path.join(stage_path, "Model.joblib")
print(f"📂 Cargando modelo desde: {model_path}")
model = joblib.load(model_path)
print(f"✅ Modelo cargado: {type(model)}")

# Cargar features
json_path = os.path.join(stage_path, "Model.json")
with open(json_path, "r") as f:
    features_and_dtypes = json.load(f)

feature_names = list(features_and_dtypes.keys())
dtypes = {col: pd.api.types.pandas_dtype(dtype) for col, dtype in features_and_dtypes.items()}

# Crear un input dummy (ajústalo con tus datos reales si quieres)
dummy_data = [{
    col: 1 if "score" not in col and "amount" not in col else 100 for col in feature_names
}]
df = pd.DataFrame.from_records(dummy_data)

# Añadir columnas faltantes
missing = set(feature_names) - set(df.columns)
for col in missing:
    df[col] = 0
df = df[feature_names].astype(dtypes)

# Hacer predicción
if hasattr(model, "predict_proba"):
    preds = model.predict_proba(df)[:, 1]
else:
    preds = model.predict(df)

print(f"📈 Predicción: {preds.tolist()}")



import joblib

model = joblib.load("/tmp/multi_model_test/init_stage/Model.joblib")
print("Pipeline:", type(model))

final_estimator = model.steps[-1][1]
print("Último paso del pipeline:", type(final_estimator))