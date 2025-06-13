import boto3
import pandas as pd
import os
import io

BUCKET_NAME = "tag-dl-sandbox-data"
S3_KEY = "output-data/test_output.pkl"

def read_pickle_from_s3():
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=BUCKET_NAME, Key=S3_KEY)
    df = pd.read_pickle(io.BytesIO(response['Body'].read()))
    print('Pickle cargado correctamente')
    return df




if __name__ == "__main__":
    df = read_pickle_from_s3()
    print(df.head(50))