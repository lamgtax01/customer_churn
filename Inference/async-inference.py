# ==== Config (edit these) ====
endpoint_name = "myAsyncEndpoint"      # your async endpoint
local_csv_path = "test.csv"            # your local test CSV in Studio
input_bucket = "your-input-bucket"     # S3 bucket you can write to
input_prefix = "async-inference/inputs"
HAS_HEADER = True                      # XGBoost expects NO header -> we'll strip if True
LABEL_IN_FIRST_COL = False             # set True if your CSV still contains label in first column
BINARY_THRESHOLD = 0.5                 # probability >= threshold -> class 1
WAIT_TIMEOUT_SEC = 600
POLL_INTERVAL_SEC = 5

# ==== Imports / clients ====
import csv, io, time, uuid, boto3
from botocore.exceptions import ClientError

session = boto3.Session()
s3 = session.client("s3")
sm = session.client("sagemaker")
smr = session.client("sagemaker-runtime")

# ==== Helpers (keep in this cell, or move to a module later) ====
def s3_uri(bucket, key): 
    return f"s3://{bucket}/{key}"

def parse_s3(s3_uri_str: str):
    assert s3_uri_str.startswith("s3://")
    no = s3_uri_str[5:]
    b, _, k = no.partition("/")
    return b, k

def object_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

def get_async_output_base(endpoint_name: str) -> str:
    """Return the S3OutputPath from the endpoint's AsyncInferenceConfig."""
    ep = sm.describe_endpoint(EndpointName=endpoint_name)
    cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
    base = cfg.get("AsyncInferenceConfig", {}).get("OutputConfig", {}).get("S3OutputPath")
    if not base:
        raise RuntimeError("Endpoint has no AsyncInferenceConfig.OutputConfig.S3OutputPath configured.")
    return base.rstrip("/")

def out_and_failure_uris(base: str, endpoint_name: str, inference_id: str):
    base_path = f"{base}/{endpoint_name}/{inference_id}"
    return f"{base_path}/out", f"{base_path}/failure"

def derive_failure_from_out(out_uri: str) -> str:
    if out_uri.endswith("/out.json"):
        return out_uri.rsplit("/out.json", 1)[0] + "/failure"
    if out_uri.endswith("/out"):
        return out_uri.rsplit("/out", 1)[0] + "/failure"
    return out_uri + ".failure"

def prep_csv_for_xgb(local_path: str, has_header: bool, drop_first_col: bool) -> bytes:
    """
    Prepare CSV for SageMaker built-in XGBoost inference:
      - remove header if present
      - drop first column if it contains the label
      - keep row order
    Returns bytes ready to upload as text/csv.
    """
    out_buf = io.StringIO()
    w = csv.writer(out_buf, lineterminator="\n")
    with open(local_path, "r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        first = True
        for row in r:
            if first and has_header:
                first = False
                continue
            first = False
            if not row:
                continue
            if drop_first_col and len(row) >= 1:
                row = row[1:]
            w.writerow(row)
    return out_buf.getvalue().encode("utf-8")

def parse_xgb_binary_output(csv_text: str, threshold: float = 0.5):
    """
    Built-in XGBoost (binary:logistic) returns one float per line in [0,1].
    Returns (labels, probs, raw_lines).
    """
    labels, probs, raws = [], [], []
    for ln in csv_text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        val = float(ln)
        probs.append(val)
        labels.append(1 if val >= threshold else 0)
        raws.append(ln)
    return labels, probs, raws

# ==== 1) Prepare and upload input CSV ====
inference_id = str(uuid.uuid4())
input_key = f"{input_prefix}/{inference_id}.csv"

body = prep_csv_for_xgb(local_csv_path, HAS_HEADER, LABEL_IN_FIRST_COL)
s3.put_object(
    Bucket=input_bucket,
    Key=input_key,
    Body=body,
    ContentType="text/csv",
)
input_s3 = s3_uri(input_bucket, input_key)
print("Uploaded CSV to:", input_s3)

# ==== 2) Invoke async endpoint ====
resp = smr.invoke_endpoint_async(
    EndpointName=endpoint_name,
    InputLocation=input_s3,
    ContentType="text/csv",
    Accept="text/csv",
    InferenceId=inference_id,
)
print("InvokeEndpointAsync ->", {k: v for k, v in resp.items() if k in ("OutputLocation", "ResponseMetadata")})

output_s3_uri = resp.get("OutputLocation")
if not output_s3_uri:
    base = get_async_output_base(endpoint_name)
    output_s3_uri, failure_s3_uri = out_and_failure_uris(base, endpoint_name, inference_id)
else:
    failure_s3_uri = derive_failure_from_out(output_s3_uri)

out_b, out_k = parse_s3(output_s3_uri)
fail_b, fail_k = parse_s3(failure_s3_uri)

# ==== 3) Poll for result or failure ====
start = time.time()
data_bytes = None
while time.time() - start < WAIT_TIMEOUT_SEC:
    if object_exists(fail_b, fail_k):
        obj = s3.get_object(Bucket=fail_b, Key=fail_k)
        err_txt = obj["Body"].read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Async inference failed:\n{err_txt}")
    if object_exists(out_b, out_k):
        obj = s3.get_object(Bucket=out_b, Key=out_k)
        data_bytes = obj["Body"].read()
        break
    time.sleep(POLL_INTERVAL_SEC)

if data_bytes is None:
    raise TimeoutError(f"Timed out after {WAIT_TIMEOUT_SEC}s waiting for output at {output_s3_uri}")

# ==== 4) Parse predictions for binary:logistic ====
text = data_bytes.decode("utf-8", errors="replace")
labels, probs, raws = parse_xgb_binary_output(text, BINARY_THRESHOLD)

print("✅ Sample predictions (first 10):")
for i in range(min(10, len(labels))):
    print(f"row {i}: label={labels[i]}  prob={probs[i]:.6f}  raw='{raws[i]}'")
