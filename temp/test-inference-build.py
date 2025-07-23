from predict import predict

data = [
    {"feature1": 0.5, "feature2": 1.0, "feature3": "value"}  # <-- ajustá a tus features reales
]

result = predict(stage="final_stage", data=data, transform=True)
print(result)