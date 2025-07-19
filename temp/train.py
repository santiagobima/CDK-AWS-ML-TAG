import pandas as pd

df = pd.read_pickle('pipelines/lead_conversion_rate/model/pickles/train.pkl')
print(df.describe())
print(df.nunique())