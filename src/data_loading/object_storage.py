import boto3
import os
import pandas as pd
import io

ENDPOINT = "https://storage.yandexcloud.net"

session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name="ru-central1"
)

s3 = session.client(
    "s3", endpoint_url=ENDPOINT)

class S3Loader():
    def __init__(self,bucket):
      self.bucket = bucket

    def get(self, filename: str) -> bytes:
        response = s3.get_object(Bucket=self.bucket, Key=filename)
        model_bytes = response['Body'].read()
        return model_bytes

def get_data(bucket_name:str, file_name:str) -> pd.DataFrame:
    """
    Получает файл из бакета

    Parameters
    ----------
    bucket_name : str
        название бакета
    file_name : str
        название файла, который мы хотим получить

    Returns
    -------
    pd.DataFrame
        Датафрейм, заполненный данными из файла
    """
    csv = S3Loader(bucket_name).get('data/'+file_name).decode('utf-8')
    return pd.read_csv(io.StringIO(csv))
