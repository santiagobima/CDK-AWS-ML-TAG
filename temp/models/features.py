import json

aws_path = "temp/models/aws_Model.json"
dp_path = "temp/models/dp_Model.json"

with open(aws_path) as f:
    aws = json.load(f)
with open(dp_path) as f:
    dp = json.load(f)

# Convierte importancias a float
aws = {k: float(v) for k, v in aws.items()}
dp = {k: float(v) for k, v in dp.items()}

features_aws = set(aws.keys())
features_dp = set(dp.keys())

print("Total features AWS:", len(aws))
print("Total features DP :", len(dp))
print("¿Features iguales?:", features_aws == features_dp)
print("Features sólo en AWS:")
print(features_aws - features_dp)
print("\nFeatures sólo en DP:")
print(features_dp - features_aws)

if features_aws == features_dp:
    diffs = {feat: abs(aws[feat] - dp[feat]) for feat in features_aws}
    max_diff = max(diffs.values())
    print("Máxima diferencia absoluta en importancia:", max_diff)
else:
    print("\nNo se puede comparar importancias porque los sets de features difieren.")