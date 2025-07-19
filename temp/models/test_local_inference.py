import os
import json
import pandas as pd

from inference import model_fn, input_fn, predict_fn, output_fn

MODEL_PATH = "temp/models"        # <--- SOLO el directorio
FEATURES_PATH = "temp/models/Model.json"

SAMPLE_INPUT = [{
    "hs_analytics_num_visits": 1,
    "hs_latest_source_DIRECT_TRAFFIC": 0.0,
    "num_conversion_events": 1,
    "n_calls_positive_outcome": 2.0,
    "hs_predictivescoringtier_tier_1": 0.0,
    "hs_email_open": 50,
    "feat_final_education_field": "economics",
    "interview_done": 0,
    "location__NA_": 0.0,
    "hs_predictivescoringtier_tier_5": 0.0,
    "hs_email_click": 0,
    "tuition_fee_amount": 6900.0,
    "hs_sa_first_engagement_object_type__NA_": 0.0,
    "average_recency_score": 0.0,
    "hs_last_sales_activity_type_MEETING_BOOKED": 20.0,
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
        if dtype == "category":
            df[col] = df[col].astype("category")
        else:
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"Warning: No se pudo convertir {col} a {dtype}: {e}")
    return df

if __name__ == "__main__":
    model_dict = model_fn(MODEL_PATH)
    print("✅ Modelo cargado.")

    input_json = json.dumps({"data": SAMPLE_INPUT})
    input_data = input_fn(input_json, "application/json")
    print("✅ Input convertido a DataFrame.")

    input_data = force_types(input_data, FEATURES_PATH)
    print("✅ Tipos de features corregidos para LightGBM.")

    try:
        prediction = predict_fn(input_data, model_dict)
        print("✅ Predicción completada:", prediction)
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        exit(2)

    output = output_fn(prediction, "application/json")
    print("✅ Salida JSON:", output)