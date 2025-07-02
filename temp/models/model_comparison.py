import joblib
import numpy as np
import os

# Ajusta la ruta según tu estructura real
MODEL_DIR = 'temp/models/final_stage/'

aws_model_path = 'temp/models/aws_final_Model.joblib'
dp_model_path  = 'temp/models/dp_final_Model.joblib'

# Cargar modelos
model_aws = joblib.load(aws_model_path)
model_dp = joblib.load(dp_model_path)

print("\n--- Hiperparámetros del modelo AWS ---")
print(model_aws.get_params())

print("\n--- Hiperparámetros del modelo DP ---")
print(model_dp.get_params())

# Si los modelos están dentro de un Pipeline, accede al modelo real así:
def get_estimator(model):
    # Busca el último step que tiene feature_importances_
    if hasattr(model, "feature_importances_"):
        return model
    elif hasattr(model, "steps"):  # sklearn Pipeline
        for name, step in model.steps[::-1]:
            if hasattr(step, "feature_importances_"):
                return step
    return model

est_aws = get_estimator(model_aws)
est_dp = get_estimator(model_dp)

print("\n--- Importancias de features (AWS) ---")
print(est_aws.feature_importances_)

print("\n--- Importancias de features (DP) ---")
print(est_dp.feature_importances_)

# ¿Son similares?
print("\n¿Son similares las importancias?")
print(np.allclose(est_aws.feature_importances_, est_dp.feature_importances_, atol=1e-5))

print("\nMáxima diferencia en importancias:")
print(np.abs(est_aws.feature_importances_ - est_dp.feature_importances_).max())

# Si tienes un set de test, puedes comparar predicciones:
# import pandas as pd
# X_test = pd.read_pickle("ruta_al_X_test.pkl")
# pred_aws = model_aws.predict_proba(X_test)[:,1]
# pred_dp = model_dp.predict_proba(X_test)[:,1]
# print("\n¿Predicciones iguales?", np.allclose(pred_aws, pred_dp, atol=1e-5))

print("\nListo. Si necesitas comparar sobre datos, sube el set de test.")