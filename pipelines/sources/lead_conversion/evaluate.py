import os
import shutil  # Importa shutil para copiar archivos
import argparse
import boto3

# Parsear los argumentos
parser = argparse.ArgumentParser()
parser.add_argument("--config_parameter", type=str, default="Developer")
parser.add_argument("--name", type=str, default="User")
parser.add_argument("--output_s3_uri", type=str, default="s3://tag-dl-sandbox-data/output-data")
parser.add_argument("--local_mode", action="store_true", help="Indica si se ejecuta en modo local")
args = parser.parse_args()

# Confirmación de ejecución
print(f"Hello {args.config_parameter}!")

# Crear el archivo de salida en el sistema de archivos local
local_output_path = "log.txt"
with open(local_output_path, "w") as f:
    f.write(f"Hello {args.config_parameter}! This script ran successfully.")

# Decidir si subir a S3 (modo local) o guardar en el sistema de SageMaker (nube)
if args.local_mode:
    # En modo local, subir el archivo directamente a S3
    s3 = boto3.client("s3")
    bucket_name, key = args.output_s3_uri.replace("s3://", "").split("/", 1)
    s3.upload_file(local_output_path, bucket_name, f"{key}/log.txt")
    print(f"Archivo guardado en {args.output_s3_uri}/log.txt en S3.")
else:
    # En la nube, copiar el archivo al directorio esperado por SageMaker
    output_dir = "/opt/ml/processing/output"
    os.makedirs(output_dir, exist_ok=True)
    # Copiar el archivo a la ubicación de salida sin problemas de sistema de archivos
    shutil.copy(local_output_path, os.path.join(output_dir, "log.txt"))
    print("Archivo guardado en el directorio de salida de SageMaker para procesamiento en la nube.")

