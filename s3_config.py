# S3 Configuration File
# Copy this file to s3_config.py and update with your actual S3 details

# S3 Bucket Configuration
S3_BUCKET_NAME = "anomaly-detect-1"

# S3 File Paths (keys)
S3_LOG_FILES = [
    "nova-sample.log",
    "anomaly_label.csv"
]

# AWS Credentials (optional - if not using AWS CLI or IAM roles)
# Leave as None to use default credential chain
AWS_ACCESS_KEY_ID = "AWS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "AWS_ACCESS_KEY"
AWS_REGION = "us-east-1"

# Local directories
LOCAL_INPUT_DIR = "./datasets"

# Enable/Disable S3 download
ENABLE_S3_DOWNLOAD = True
