from datetime import datetime

# Initialize global trackers
total_value = 0
unique_columns = set()
unique_file_paths = set()
unique_values = []  # Final list of lists
unique_indexed_dict = {}  # Intermediate dict to deduplicate and aggregate

# Define the customer ID pattern (already provided)
customer_id_regex = re.compile(r'^1000\d{8}$')

def extract_matching_values(df, bucket, key):
    global total_value
    matched = False
    file_uri = f"{bucket}/{key}"
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    bucket_suffix_map = {
        'use1-afa1': 'bucket1',
        'use1-afa2': 'bucket2',
        'usw2-afa1': 'bucket3',
        'usw2-afa2': 'bucket4'
    }

    bucket_column = None
    for suffix, colname in bucket_suffix_map.items():
        if bucket.endswith(suffix):
            bucket_column = colname
            break

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
                    val = str(val).strip()

                    if val not in unique_indexed_dict:
                        # Initialize record
                        new_record = {
                            'id': val,
                            'date': now,
                            'bucket1': '',
                            'bucket2': '',
                            'bucket3': '',
                            'bucket4': '',
                            's3_uris': set(),
                            'columns': set()
                        }
                        unique_indexed_dict[val] = new_record

                    record = unique_indexed_dict[val]

                    if bucket_column:
                        record[bucket_column] = bucket

                    record['s3_uris'].add(file_uri)
                    record['columns'].add(col)

        except Exception as e:
            continue

    return matched


# Convert intermediate dict to final list of lists
unique_values = [
    [
        val,
        record['date'],
        record['bucket1'],
        record['bucket2'],
        record['bucket3'],
        record['bucket4'],
        sorted(list(record['s3_uris'])),
        sorted(list(record['columns']))
    ]
    for val, record in unique_indexed_dict.items()
]


import pandas as pd
import io

def write_unique_values_to_csv():
    df = pd.DataFrame(
        unique_values,
        columns=['id', 'date', 'bucket1', 'bucket2', 'bucket3', 'bucket4', 's3_url', 'columns']
    )

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.write(f"\ntotalValue : {total_value}")
    buffer.write(f"\ntotalUniqueValue : {len(unique_values)}")
    s3.put_object(Bucket=target_bucket, Key=f"{output_prefix}unique_file.csv", Body=buffer.getvalue().encode("utf-8"))
