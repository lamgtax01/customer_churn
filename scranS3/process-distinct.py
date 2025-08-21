def extract_matching_duplicated_values(df, bucket, key, obj_metadata):
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    file_uri = f"{bucket}/{key}"

    # Extract S3 object metadata
    file_owner = obj_metadata.get('Owner', {}).get('DisplayName', '')
    last_modified = obj_metadata.get('LastModified', '')
    storage_class = obj_metadata.get('StorageClass', '')
    file_size = obj_metadata.get('ContentLength', '')
    content_type = obj_metadata.get('ContentType', '')
    file_type = infer_file_type_from_key(key)

    for col in df.columns:
        try:
            col_series = df[col].astype(str)
            matches = col_series[col_series.str.match(customer_id_regex)]
            if not matches.empty:
                for val in matches:
                    val = str(val).strip()
                    duplicated_values.append([
                        val,
                        now,
                        bucket,
                        file_uri,
                        col,
                        file_owner,
                        last_modified,
                        file_size,
                        file_type,
                        storage_class
                    ])
        except Exception:
            continue

import pyarrow.parquet as pq
import io

def read_parquet_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj['Body'].read()
    table = pq.read_table(io.BytesIO(data))
    return table.to_pandas()