import boto3
import pandas as pd
import io
import json
import re
import pyarrow.parquet as pq
import pyarrow.fs as fs
from urllib.parse import urlparse

# --- Config ---
buckets_to_scan = ["bucketA", "bucketB1", "bucketB2", "bucketB3", "bucketB4"]
target_bucket = "bucketB1"  # Where to save output
output_prefix = "preprocessing_output/"
customer_id_regex = re.compile(r'^1000\d{8}$')

# --- AWS Clients ---
s3 = boto3.client('s3')

# --- Trackers ---
total_value = 0
unique_values = {}
unique_columns = set()
unique_file_paths = set()

def list_all_files(bucket):
    paginator = s3.get_paginator('list_objects_v2')
    all_keys = []
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get('Contents', []):
            all_keys.append(obj['Key'])
    return all_keys

def extract_matching_values(df, bucket, key):
    global total_value
    matched = False
    file_uri = f"{bucket}/{key}"
    for col in df.columns:
        try:
            col_series = df[col].astype(str)
            matches = col_series[col_series.str.match(customer_id_regex)]
            if not matches.empty:
                matched = True
                unique_columns.add(col)
                unique_file_paths.add(file_uri)
                for val in matches:
                    total_value += 1
                    if val not in unique_values:
                        unique_values[val] = set()
                    unique_values[val].add(file_uri)
        except Exception:
            continue
    return matched

def process_file(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj['Body'].read()
        if key.endswith(".csv") or key.endswith(".txt"):
            df = pd.read_csv(io.BytesIO(body), dtype=str, low_memory=False)
            extract_matching_values(df, bucket, key)
        elif key.endswith(".json"):
            df = pd.read_json(io.BytesIO(body), lines=True)
            extract_matching_values(df, bucket, key)
        elif key.endswith(".parquet"):
            uri = f"s3://{bucket}/{key}"
            df = pd.read_parquet(uri, engine='pyarrow')
            extract_matching_values(df, bucket, key)
    except Exception as e:
        print(f"⚠️ Failed to process {bucket}/{key}: {e}")

def write_and_upload_csv(dataframe, bucket, key_name):
    buffer = io.StringIO()
    dataframe.to_csv(buffer, index=False)
    buffer.write(f"\ntotalValue : {total_value}")
    buffer.write(f"\ntotalUniqueValue : {len(unique_values)}")
    s3.put_object(Bucket=bucket, Key=key_name, Body=buffer.getvalue().encode("utf-8"))
    print(f"✅ Uploaded: s3://{bucket}/{key_name}")

def main():
    print("🔍 Starting scan...")
    for bucket in buckets_to_scan:
        print(f"📁 Scanning bucket: {bucket}")
        keys = list_all_files(bucket)
        for key in keys:
            if key.endswith((".csv", ".json", ".parquet", ".txt")):
                process_file(bucket, key)

    print("🧠 Building output files...")

    # unique_file.csv
    data = []
    for customer_id, uris in unique_values.items():
        data.append({
            "customer_id": customer_id,
            "S3 URI": json.dumps(sorted(list(uris)))
        })
    df_unique_file = pd.DataFrame(data)
    write_and_upload_csv(df_unique_file, target_bucket, f"{output_prefix}unique_file.csv")

    # unique_columns.csv
    df_columns = pd.DataFrame(sorted(unique_columns), columns=["column_name"])
    write_and_upload_csv(df_columns, target_bucket, f"{output_prefix}unique_columns.csv")

    # unique_file_path.csv
    df_paths = pd.DataFrame(sorted(unique_file_paths), columns=["S3 URI"])
    write_and_upload_csv(df_paths, target_bucket, f"{output_prefix}unique_file_path.csv")

    # Final stats
    print(f"🔢 totalValue: {total_value}")
    print(f"🔢 totalUniqueValue: {len(unique_values)}")

if __name__ == "__main__":
    main()
