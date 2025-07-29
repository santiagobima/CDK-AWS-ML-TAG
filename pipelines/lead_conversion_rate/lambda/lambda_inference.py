import json
import boto3
import os
import tarfile
import joblib
import pandas as pd
import traceback
from sklearn.pipeline import Pipeline

print("🚀 Ejecutando lambda_inference.py")

s3 = boto3.client("s3")


def lambda_handler(event, context):
    try:
        print("📥 Evento recibido:", event)

        # Si viene desde API Gateway con estructura envolvente
        if isinstance(event, dict) and "body" in event:
            body = event["body"]
            if isinstance(body, str):
                body = json.loads(body)
        else:
            body = event

        stage = body.get("stage", "init_stage")
        data = body.get("data", [])

        if not data:
            raise ValueError("El campo 'data' está vacío o no proporcionado.")

        bucket = os.environ["MODEL_BUCKET"]
        key = "output-data/predict/models/tar_models/multiendpoint.tar.gz"
        local_tar_path = "/tmp/multiendpoint.tar.gz"
        stage_path = f"/tmp/{stage}"

        print(f"📂 Stage: {stage}")
        print(f"🎯 S3 path: s3://{bucket}/{key}")
        print(f"📍 Ruta local del stage: {stage_path}")

        # Descargar y extraer si no existe ya
        if not os.path.exists(stage_path):
            print("⬇️ Descargando modelo...")
            s3.download_file(bucket, key, local_tar_path)

            print("📦 Extrayendo modelo...")
            with tarfile.open(local_tar_path, "r:gz") as tar:
                tar.extractall(path="/tmp")
            print("✅ Extracción completa")

        # Cargar modelo
        model_path = os.path.join(stage_path, "Model.joblib")
        print(f"🔍 Cargando modelo desde: {model_path}")
        model = joblib.load(model_path)
        print(f"✅ Modelo cargado: {type(model)}")

        if isinstance(model, Pipeline):
            print(f"🔍 Último paso del pipeline: {type(model.steps[-1][1])}")
        else:
            print("⚠️ Modelo NO es un Pipeline. Esto puede causar errores de predicción.")

        # Cargar features esperados y tipos
        features_path = os.path.join(stage_path, "Model.json")
        print(f"📄 Cargando features desde: {features_path}")
        with open(features_path, "r") as f:
            features_and_dtypes = json.load(f)

        feature_names = list(features_and_dtypes.keys())
        dtypes = {col: pd.api.types.pandas_dtype(dtype) for col, dtype in features_and_dtypes.items()}
        print(f"📊 Features esperados: {feature_names}")

        # Preparar DataFrame
        df = pd.DataFrame.from_records(data)
        print("🧹 Preparando DataFrame...")

        for col in feature_names:
            if col not in df.columns:
                print(f"⚠️ Feature faltante: {col} — se rellena con 0")
                df[col] = 0

        df = df[feature_names].astype(dtypes)
        print("✅ DataFrame listo para predecir")

        # Predicción
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(df)[:, 1].tolist()
        else:
            preds = model.predict(df).tolist()

        print("✅ Predicción completada")
        return {
            "statusCode": 200,
            "body": json.dumps({"predictions": preds})
        }

    except Exception as e:
        print("❌ Excepción capturada:")
        print(traceback.format_exc())

        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        }