import os

BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
OBJECT_KEY = os.environ.get('S3_OBJECT_KEY', 'cookie/model.pth')
MODEL_LOCAL_FILE_PATH = "model.pth"