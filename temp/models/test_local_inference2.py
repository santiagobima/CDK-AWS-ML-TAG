import os
import json
import pandas as pd
from inference import model_fn, input_fn, predict_fn, output_fn

MODEL_PATH = "temp/models"  # donde están Model.joblib y Model.json

SAMPLE_INPUT = [{
    "hs_analytics_num_visits": 1,
    "hs_latest_source_DIRECT_TRAFFIC": 0.0,
    "num_conversion_events": 1,
    "n_calls_positive_outcome": 0.0,
    "hs_predictivescoringtier_tier_1": 0.0,
    "hs_email_open": 5,
    "feat_final_education_field": "economics",
    "interview_done": 0,
    "location__NA_": 0.0,
    "hs_predictivescoringtier_tier_5": 0.0,
    "hs_email_click": 0,
    "tuition_fee_amount": 6900.0,
    "hs_sa_first_engagement_object_type__NA_": 0.0,
    "average_recency_score": 0.0,
    "hs_last_sales_activity_type_MEETING_BOOKED": 0.0,
    "hs_last_sales_activity_type__NA_": 0.0,
    "hs_email_sends_since_last_engagement": 0,
    "hs_latest_source_PAID_SOCIAL": 1.0,
    "amount": 6900.0,
    "sku_tuition": "ITDMFT1915"
}]

def force_types(df, features_path):
    with open(features_path, "r") as f:
        feature_types = json.load(f)
    for col, dtype in feature_types.items():
        if col in df.columns:
            if dtype == "category":
                df[col] = df[col].astype("category")
            else:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: {col} no pudo convertirse a {dtype}: {e}")
        else:
            df[col] = 0  # Default para columnas faltantes
    return df[list(feature_types.keys())]

if __name__ == "__main__":
    model_dict = model_fn(MODEL_PATH)
    print("✅ Modelo cargado.")

    input_json = json.dumps({"data": SAMPLE_INPUT})
    input_data = input_fn(input_json, "application/json")
    print("✅ Input convertido a DataFrame.")

    input_data = force_types(input_data, os.path.join(MODEL_PATH, "Model.json"))
    print("✅ Tipos de features corregidos.")

    try:
        prediction = predict_fn(input_data, model_dict)
        print("✅ Predicción completada:", prediction)
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        exit(2)

    output = output_fn(prediction, "application/json")
    print("✅ Salida JSON:", output)