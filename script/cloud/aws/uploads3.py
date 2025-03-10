import argparse
import os

import boto3


def upload_dir(target_dir, bucket, prefix=""):
    s3 = boto3.client("s3")
    for root, _, files in os.walk(target_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, target_dir)
            key = os.path.join(prefix, rel_path) if prefix else rel_path
            s3.upload_file(local_path, bucket, key)
            print(f"uploaded {local_path} to s3://{bucket}/{key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="recursively upload a dir to s3")
    parser.add_argument("target_dir", help="directory to upload")
    parser.add_argument("bucket", help="s3 bucket name")
    parser.add_argument("--prefix", default="", help="optional s3 key prefix")
    args = parser.parse_args()
    upload_dir(args.target_dir, args.bucket, args.prefix)
