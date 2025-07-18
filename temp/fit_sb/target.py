import os
import pandas as pd
from sklearn.model_selection import train_test_split

from pipelines.lead_conversion_rate.steps.predict import predict

# Ruta base
base_path = "temp/fit_sb"
file_path = os.path.join(base_path, "baseline_features_raw.pkl")

# Cargar dataset
df = pd.read_pickle(file_path)

# Verificar que 'target' esté presente
if "target" not in df.columns:
    raise ValueError(f"La columna 'target' no se encuentra en el archivo {file_path}. "
                     f"Las columnas disponibles son: {list(df.columns)}")

# Separar features y target
X = df.drop(columns=["target"])
y = df["target"]

# Split en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar test sets en la misma carpeta
X_test_path = os.path.join(base_path, "X_test.pkl")
y_test_path = os.path.join(base_path, "y_test.pkl")

X_test.to_pickle(X_test_path)
y_test.to_pickle(y_test_path)

# Cargar X_test desde el path guardado
X_test_loaded = pd.read_pickle(X_test_path)
X_test_dict = X_test_loaded.to_dict(orient="records")

# Ejecutar predicción para cada stage
preds_init = predict("init_stage", data=X_test_dict)
preds_mid = predict("mid_stage", data=X_test_dict)
preds_final = predict("final_stage", data=X_test_dict)

# Mostrar resultados
print("Predicciones etapa init:", preds_init)
print("Predicciones etapa mid:", preds_mid)
print("Predicciones etapa final:", preds_final)

# Ground truth
y_test_loaded = pd.read_pickle(y_test_path)
print("Ground truth:", y_test_loaded.values)