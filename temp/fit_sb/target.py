import pandas as pd

df = pd.read_pickle("temp/fit_sb/baseline_features_raw.pkl")
print(df.columns)
print(df.head())