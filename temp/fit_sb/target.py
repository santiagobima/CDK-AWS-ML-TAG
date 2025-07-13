import pandas as pd

df = pd.read_pickle("temp/fit_sb/X_test.pkl")
print(df.columns)
print(df.head())