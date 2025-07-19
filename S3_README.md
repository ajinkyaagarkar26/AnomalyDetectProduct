# S3 Integration for Log Anomaly Detection

This project now supports downloading log files from Amazon S3 before processing them for anomaly detection.

## Setup

### 1. Install Dependencies

Make sure you have boto3 installed:
```bash
pip install boto3
```

Or install all requirements:
```bash
pip install -r requirements/requirements.txt
```

### 2. Configure AWS Credentials

You can configure AWS credentials in several ways:

#### Option A: AWS CLI (Recommended)
```bash
aws configure
```

#### Option B: Environment Variables
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

#### Option C: IAM Roles (for EC2 instances)
If running on EC2, you can use IAM roles attached to the instance.

### 3. Configure S3 Settings

1. Copy the configuration template:
   ```bash
   cp s3_config_template.py s3_config.py
   ```

2. Edit `s3_config.py` with your S3 details:
   ```python
   # S3 Bucket Configuration
   S3_BUCKET_NAME = "your-actual-bucket-name"
   
   # S3 File Paths (keys)
   S3_LOG_FILES = [
       "path/to/your/nova-sample.log",
       "path/to/your/anomaly_label.csv"
   ]
   
   # Enable S3 download
   ENABLE_S3_DOWNLOAD = True
   ```

## Usage

### Running with S3 Download

1. Make sure your S3 configuration is set up correctly in `s3_config.py`
2. Set `ENABLE_S3_DOWNLOAD = True` in your configuration
3. Run the data processing script:
   ```bash
   python data_process.py
   ```

The script will:
1. Download the specified files from S3 to the `./datasets` directory
2. Proceed with the normal log parsing and anomaly detection workflow

### Running without S3 (Default)

If you don't need S3 integration:
1. Keep `ENABLE_S3_DOWNLOAD = False` (default)
2. Ensure your log files are already in the `./datasets` directory
3. Run the script normally

## S3 Utils Module

The `s3_utils.py` module provides a reusable `S3Utils` class with the following features:

- Download single or multiple files from S3
- Automatic directory creation
- Comprehensive error handling
- List bucket contents
- Support for custom AWS credentials

### Example Usage

```python
from s3_utils import S3Utils, download_log_files_from_s3

# Quick download
results = download_log_files_from_s3(
    bucket_name="my-log-bucket",
    s3_keys=["logs/nova-sample.log", "logs/anomaly_label.csv"],
    local_dir="./datasets"
)

# Advanced usage
s3_utils = S3Utils(region_name="us-west-2")
success = s3_utils.download_file(
    bucket_name="my-bucket",
    s3_key="logs/application.log",
    local_file_path="./data/application.log"
)
```

## Troubleshooting

### Common Issues

1. **Credentials Error**: Make sure AWS credentials are properly configured
2. **Bucket Not Found**: Verify the bucket name in your configuration
3. **File Not Found**: Check that the S3 keys (file paths) are correct
4. **Permission Denied**: Ensure your AWS user/role has S3 read permissions

### Error Messages

- `AWS credentials not found`: Configure AWS credentials
- `Bucket 'name' does not exist`: Check bucket name spelling
- `Object 'key' does not exist`: Verify the S3 file path
- `Failed to download`: Check network connectivity and permissions

## File Structure

```
├── s3_utils.py              # S3 utility functions
├── s3_config_template.py    # Configuration template
├── s3_config.py            # Your actual configuration (create this)
├── data_process.py         # Main processing script (updated)
└── S3_README.md           # This documentation
```
