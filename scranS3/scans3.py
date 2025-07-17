import boto3
import pandas as pd
import io
import re
import pyarrow.parquet as pq
import pyarrow.fs as fs
from urllib.parse import urlparse

# CONFIG
buckets_to_scan = ["bucketA", "bucketB1", "bucketB2", "bucketB3", "bucketB4"]
target_bucket = "bucketB1"
target_key = "unique_file.csv"

# Initialize
s3 = boto3.client('s3')
total_value = 0
unique_values = set()
results = []

# Regex for identifying customer ID pattern
customer_id_pattern = re.compile(r'^1000\d{8}$')  # 12-digit, starts with 1000

def list_all_files(bucket):
    paginator = s3.get_paginator('list_objects_v2')
    keys = []
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get('Contents', []):
            keys.append(obj['Key'])
    return keys

def extract_matching_column(df, bucket, key):
    global total_value
    file_name = key.split("/")[-1]
    s3_prefix = "/".join(key.split("/")[:-1])
    
    for col in df.columns:
        try:
            matched_values = df[col].astype(str).str.extract(f'({customer_id_pattern.pattern})').dropna()[0]
            if not matched_values.empty:
                for val in matched_values:
                    total_value += 1
                    if val not in unique_values:
                        unique_values.add(val)
                        results.append({
                            'customer_id': val,
                            'column_name': col,
                            'bucket': bucket,
                            'file_name': file_name,
                            's3_prefix': s3_prefix
                        })
                break  # Only one column per file
        except Exception as e:
            continue

def process_file(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        if key.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(obj['Body'].read()), dtype=str)
            extract_matching_column(df, bucket, key)

        elif key.endswith(".json"):
            df = pd.read_json(io.BytesIO(obj['Body'].read()), lines=True)
            extract_matching_column(df, bucket, key)

        elif key.endswith(".parquet"):
            # Read parquet using Arrow
            uri = f"s3://{bucket}/{key}"
            df = pd.read_parquet(uri, engine='pyarrow')
            extract_matching_column(df, bucket, key)

    except Exception as e:
        print(f"Failed to process {bucket}/{key}: {e}")

def main():
    for bucket in buckets_to_scan:
        print(f"🔍 Scanning bucket: {bucket}")
        try:
            keys = list_all_files(bucket)
            for key in keys:
                if key.endswith((".csv", ".json", ".parquet")):
                    process_file(bucket, key)
        except Exception as e:
            print(f"Error in bucket {bucket}: {e}")

    # Save to CSV
    df_result = pd.DataFrame(results)
    csv_buffer = io.StringIO()
    df_result.to_csv(csv_buffer, index=False)

    # Add summary lines
    csv_buffer.write(f"\ntotalValue : {total_value}")
    csv_buffer.write(f"\ntotalUniqueValue : {len(unique_values)}")

    # Upload to S3
    s3.put_object(Bucket=target_bucket, Key=target_key, Body=csv_buffer.getvalue().encode('utf-8'))

    print(f"\n✅ Uploaded result to s3://{target_bucket}/{target_key}")
    print(f"🔢 totalValue: {total_value}")
    print(f"🔢 totalUniqueValue: {len(unique_values)}")

if __name__ == "__main__":
    main()
