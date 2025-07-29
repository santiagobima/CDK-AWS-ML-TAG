import boto3
import tarfile
import os
import joblib

# Configura tus valores
bucket = "tag-dl-sandbox-data"
key = "output-data/predict/models/tar_models/multiendpoint.tar.gz"
local_tar_path = "/tmp/multiendpoint.tar.gz"
extract_dir = "/tmp/multi_models"

# Limpia si ya existe
if os.path.exists(extract_dir):
    import shutil
    shutil.rmtree(extract_dir)
os.makedirs(extract_dir, exist_ok=True)

# Descargar desde S3
s3 = boto3.client("s3")
print(f"📥 Descargando {key} desde S3...")
s3.download_file(bucket, key, local_tar_path)
print("✅ Descargado.")

# Extraer
print("📦 Extrayendo tar.gz...")
with tarfile.open(local_tar_path, "r:gz") as tar:
    tar.extractall(path=extract_dir)
print("✅ Extraído en:", extract_dir)

# Revisar cada modelo
for stage in ["init_stage", "mid_stage", "final_stage"]:
    model_path = os.path.join(extract_dir, stage, "Model.joblib")
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo para {stage}")
        continue

    try:
        model = joblib.load(model_path)
        print(f"✅ Modelo {stage} cargado exitosamente: {type(model)}")
    except Exception as e:
        print(f"❌ Error al cargar modelo {stage}: {e}")