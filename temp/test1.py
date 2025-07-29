import boto3
import json

bucket_name = "tag-dl-sandbox-data"
key = "output-data/predict/models/init_stage/Model.json"

s3 = boto3.client("s3")
response = s3.get_object(Bucket=bucket_name, Key=key)
content = response['Body'].read().decode('utf-8')

features = json.loads(content)

print("📌 Features encontrados:")
print(json.dumps(features, indent=2))