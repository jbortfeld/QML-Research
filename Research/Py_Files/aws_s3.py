import boto3


def s3_check_file_exists(bucket_name:str='qml-solutions-new-york', 
                         file_key:str='/factset-api-global-prices/B01DPB-R.csv', 
                         aws_access_key_id:str=None, 
                         aws_secret_access_key:str=None):
    
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    try:
        s3.head_object(Bucket=bucket_name, Key=file_key)
        return True
    except s3.exceptions.ClientError:
        return False

def copy_file_to_s3(local_file_path:str=None, 
                     s3_bucket:str='qml-solutions-new-york', 
                     s3_key:str='factset-api-global-prices/', 
                     aws_access_key_id:str=None, 
                     aws_secret_access_key:str=None,
                     verbose:bool=False):
    
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )   
    try:
        s3.upload_file(local_file_path, s3_bucket, s3_key)
        if verbose:
            print(f'--uploaded {local_file_path} to {s3_bucket}/{s3_key}')
        return True
    except:
        return False


def list_s3_bucket_contents(bucket_name, prefix='', aws_access_key_id=None, aws_secret_access_key=None):
    """
    List all items in an S3 bucket and subfolder.
    
    Parameters:
    - bucket_name: str, name of the S3 bucket
    - prefix: str, the folder path within the bucket (optional)
    
    Returns:
    - List of file keys (paths) in the specified bucket and folder
    """
    s3_client = boto3.client('s3', 
                             aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,)
    paginator = s3_client.get_paginator('list_objects_v2')
    
    file_keys = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_keys.append(obj['Key'])
    
    return file_keys


def delete_s3_contents(bucket_name, prefix="",aws_access_key_id:str=None, 
                         aws_secret_access_key:str=None):
    """
    Deletes all objects in an S3 bucket or within a specified prefix (folder).
    
    :param bucket_name: Name of the S3 bucket.
    :param prefix: (Optional) Prefix (folder path) to delete objects within.
    """
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )   

    # List objects in the specified bucket and prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if 'Contents' not in response:
        print("No objects found.")
        return
    

    # Extract object keys
    objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
    

    # Delete objects
    s3.delete_objects(Bucket=bucket_name, Delete={'Objects': objects_to_delete})

    print(f"Deleted {len(objects_to_delete)} objects from {bucket_name}/{prefix}")
