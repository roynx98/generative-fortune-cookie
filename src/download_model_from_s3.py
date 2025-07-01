import boto3
from config import BUCKET_NAME, OBJECT_KEY, MODEL_LOCAL_FILE_PATH

def download_model_from_s3():
    s3 = boto3.client("s3")
    try:
        print("Downloading from S3...")
        s3.download_file(BUCKET_NAME, OBJECT_KEY, MODEL_LOCAL_FILE_PATH)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download file: {e}")
        exit(1)
    
download_model_from_s3()