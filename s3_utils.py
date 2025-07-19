import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError


class S3Utils:
    """Utility class for S3 operations"""
    
    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, region_name='us-east-1'):
        """
        Initialize S3 client
        
        Args:
            aws_access_key_id (str, optional): AWS access key ID
            aws_secret_access_key (str, optional): AWS secret access key
            region_name (str): AWS region name (default: us-east-1)
        """
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name
                )
            else:
                # Use default credentials (from AWS CLI, environment variables, or IAM role)
                self.s3_client = boto3.client('s3', region_name=region_name)
        except Exception as e:
            print(f"Error initializing S3 client: {e}")
            raise
    
    def download_file(self, bucket_name, s3_key, local_file_path):
        """
        Download a file from S3 to local storage
        
        Args:
            bucket_name (str): Name of the S3 bucket
            s3_key (str): S3 object key (path to file in bucket)
            local_file_path (str): Local path where file should be saved
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Download file
            print(f"Downloading {s3_key} from bucket {bucket_name} to {local_file_path}...")
            self.s3_client.download_file(bucket_name, s3_key, local_file_path)
            print(f"Successfully downloaded {s3_key} from bucket {bucket_name} to {local_file_path}")
            return True
            
        except NoCredentialsError:
            print("Error: AWS credentials not found")
            return False
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                print(f"Error: Bucket '{bucket_name}' does not exist")
            elif error_code == 'NoSuchKey':
                print(f"Error: Object '{s3_key}' does not exist in bucket '{bucket_name}'")
            else:
                print(f"Error downloading file: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
    
    def download_multiple_files(self, downloads):
        """
        Download multiple files from S3
        
        Args:
            downloads (list): List of tuples (bucket_name, s3_key, local_file_path)
            
        Returns:
            dict: Dictionary with download results {local_file_path: success_status}
        """
        results = {}
        for bucket_name, s3_key, local_file_path in downloads:
            success = self.download_file(bucket_name, s3_key, local_file_path)
            results[local_file_path] = success
        return results
    
    def list_bucket_contents(self, bucket_name, prefix=''):
        """
        List contents of an S3 bucket
        
        Args:
            bucket_name (str): Name of the S3 bucket
            prefix (str): Prefix to filter objects (optional)
            
        Returns:
            list: List of object keys in the bucket
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            else:
                print(f"No objects found in bucket '{bucket_name}' with prefix '{prefix}'")
                return []
                
        except ClientError as e:
            print(f"Error listing bucket contents: {e}")
            return []


def download_log_files_from_s3(bucket_name=None, s3_keys=None, local_dir='./datasets', use_credentials=True):
    """
    Convenience function to download log files from S3
    Uses configuration from s3_config.py if parameters are not provided
    
    Args:
        bucket_name (str, optional): Name of the S3 bucket (uses config if None)
        s3_keys (list or str, optional): S3 object key(s) for the log file(s) (uses config if None)
        local_dir (str): Local directory to save files (default: ./datasets)
        use_credentials (bool): Whether to use explicit credentials from config (default: True)
        
    Returns:
        dict: Dictionary with download results {local_file_path: success_status}
    """
    # Try to import configuration if parameters not provided
    if bucket_name is None or s3_keys is None:
        try:
            from s3_config import S3_BUCKET_NAME, S3_LOG_FILES, LOCAL_INPUT_DIR
            if bucket_name is None:
                bucket_name = S3_BUCKET_NAME
            if s3_keys is None:
                s3_keys = S3_LOG_FILES
            if local_dir == './datasets':  # Use config dir if using default
                local_dir = LOCAL_INPUT_DIR
        except ImportError:
            raise ValueError("s3_config.py not found and bucket_name/s3_keys not provided. "
                           "Either provide parameters or create s3_config.py from template.")
    
    # Initialize S3Utils with or without explicit credentials
    if use_credentials:
        try:
            from s3_config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
            s3_utils = S3Utils(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
        except ImportError:
            print("Warning: s3_config.py not found, using default credential chain")
            s3_utils = S3Utils()
    else:
        # Use default credential chain (IAM roles, environment variables, AWS CLI config)
        s3_utils = S3Utils()
    
    # Convert single key to list
    if isinstance(s3_keys, str):
        s3_keys = [s3_keys]
    
    downloads = []
    for s3_key in s3_keys:
        # Extract filename from S3 key
        filename = os.path.basename(s3_key)
        local_file_path = os.path.join(local_dir, filename)
        downloads.append((bucket_name, s3_key, local_file_path))
    
    return s3_utils.download_multiple_files(downloads)


if __name__ == "__main__":
    # Example usage - now uses configuration from s3_config.py
    try:
        from s3_config import S3_BUCKET_NAME, S3_LOG_FILES, LOCAL_INPUT_DIR
        
        # Download files using configuration
        print("Downloading files using s3_config.py...")
        download_results = download_log_files_from_s3()
        
        for file_path, success in download_results.items():
            if success:
                print(f"✓ Successfully downloaded: {file_path}")
            else:
                print(f"✗ Failed to download: {file_path}")
                
    except ImportError:
        print("s3_config.py not found. Using manual configuration...")
        # Fallback to manual configuration
        bucket_name = "your-log-bucket"
        s3_key = "logs/nova-sample.log"
        local_file_path = "./datasets/nova-sample.log"
        
        s3_utils = S3Utils()
        success = s3_utils.download_file(bucket_name, s3_key, local_file_path)
        
        if success:
            print("File downloaded successfully!")
        else:
            print("File download failed!")
