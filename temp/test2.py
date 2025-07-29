# temp/verify_tar_contents.py
import tarfile
import boto3
import tempfile
import joblib
import os

s3 = boto3.client("s3")
bucket = "your-bucket-name"
key = "output-data/predict/models/tar_models/multiendpoint.tar.gz"

with tempfile.TemporaryDirectory() as tmpdir:
    tar_path = os.path.join(tmpdir, "multi.tar.gz")
    s3.download_file(bucket, key, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=tmpdir)

    for stage in ["init_stage", "mid_stage", "final_stage"]:
        model_path = os.path.join(tmpdir, stage, "Model.joblib")
        print(f"🔍 Revisando modelo: {model_path}")
        model = joblib.load(model_path)
        print(f"✅ Modelo {stage} es de tipo: {type(model)}")