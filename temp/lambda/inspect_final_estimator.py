import joblib

model_path = "/tmp/multi_model_test/init_stage/Model.joblib"
model = joblib.load(model_path)

print("✅ Modelo cargado:", type(model))

final_estimator = model.steps[-1][1]
print("🧠 Último paso del pipeline:", type(final_estimator))