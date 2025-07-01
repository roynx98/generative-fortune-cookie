import boto3
from config import BUCKET_NAME, OBJECT_KEY, MODEL_LOCAL_FILE_PATH

def upload_model_to_s3():
    s3 = boto3.client("s3")
    s3.upload_file(MODEL_LOCAL_FILE_PATH, BUCKET_NAME, OBJECT_KEY)

upload_model_to_s3()