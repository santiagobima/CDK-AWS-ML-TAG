import boto3
import pandas as pd
import joblib
import json
import io
import os

s3 = boto3.client("s3")

BUCKET = os.getenv("MODEL_BUCKET")
MODEL_PREFIX = "output-data/predict/models"

def load_model_from_s3(stage):
    key = f"{MODEL_PREFIX}/{stage}/Model.joblib"
    response = s3.get_object(Bucket=BUCKET, Key=key)
    return joblib.load(io.BytesIO(response['Body'].read()))

def load_features_from_s3(stage):
    key = f"{MODEL_PREFIX}/{stage}/Model.json"
    response = s3.get_object(Bucket=BUCKET, Key=key)
    return json.loads(response['Body'].read())

def preprocess_input(data, features_types):
    df = pd.DataFrame(data)
    for col, dtype in features_types.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
        else:
            df[col] = 0
    return df[list(features_types.keys())]

def predict(model, df):
    return model.predict_proba(df)[:, 1]