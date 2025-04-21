import os
import boto3
import io
import csv
from datetime import datetime

# --- 0. ENVIRONMENT VARIABLES ---
S3_BUCKET_OUTPUT = os.environ["S3_BUCKET_OUTPUT"]   
S3_KPI_KEY = os.environ["S3_KPI_KEY"]               
S3_KPI_FOLDER = os.environ["S3_KPI_FOLDER"]               
SART_DATE = os.environ["SART_DATE"]                 
SNS_TOPIC = os.environ["SNS_TOPIC"]                 
KPI_THRESHOLD = float(os.environ["KPI_THRESHOLD"])  

print("S3_BUCKET_OUTPUT:", S3_BUCKET_OUTPUT)
print("S3_KPI_KEY:", S3_KPI_KEY)
print("SART_DATE:", SART_DATE)
print("SNS_TOPIC:", SNS_TOPIC)
print("KPI_THRESHOLD:", KPI_THRESHOLD)

# --- 1. READ CSV FROM S3 ---
s3 = boto3.client("s3")
sns = boto3.client("sns")

REGION = boto3.session.Session().region_name
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

def lambda_handler(event, context):
    response = s3.get_object(Bucket=S3_BUCKET_OUTPUT, Key=S3_KPI_KEY)
    content = response["Body"].read().decode("utf-8-sig")
    csv_reader = csv.reader(io.StringIO(content))

    # --- 2. FILTER VIOLATIONS ---
    violations = []
    for row in csv_reader:
        print(row)
        try:
            name = row[1]  
            csi = float(row[2])     
            date = str(row[3])              
            if date > SART_DATE and csi > KPI_THRESHOLD:
                print("name:",name, "psi:", csi , "date:", date)
                violations.append({"name": name, "csi": csi, "date": date})
        except (KeyError, ValueError) as e:
            continue  

    # --- 3. FORMAT SNS MESSAGE ---
    if not violations:
        message = "✅ No model drift detected. All CSI values are within threshold."
        subject = "✅ Model Drift Monitoring Alert"
    else:
        table = "| Feature | PSI | Date |\n|---------|-----|------|\n"
        drift_csv_s3_uri = None
        # Write violations to in-memory CSV
        output_buffer = io.StringIO()
        writer = csv.writer(output_buffer)
        writer.writerow(["name", "csi", "date"])
        for v in violations:
            table += f"| {v['name']} | {v['csi']} | {v['date']} |\n"
            writer.writerow([v["name"], v["csi"], v["date"]])

        # Prepare S3 path
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")  
        drift_key = f"{S3_KPI_FOLDER}/drifts-detected/{date_str}/drifts_detected_csi.csv"

        # Upload to S3
        s3.put_object(
            Bucket=S3_BUCKET_OUTPUT,
            Key=drift_key,
            Body=output_buffer.getvalue()
        )

        drift_csv_s3_uri = f"s3://{S3_BUCKET_OUTPUT}/{drift_key}"
        print("Uploaded drift report to:", drift_csv_s3_uri)

        subject="🚨 Model Drift Monitoring Alert"
        message = f"""⚠️ **Model Drift Detected!**\n\n
                   📎Full violation CSV: {drift_csv_s3_uri} 
                   The following features have KPI > {KPI_THRESHOLD} after {SART_DATE}:\n\n
                   {table}"""

    # --- 4. SEND SNS NOTIFICATION ---
    sns.publish(
        TopicArn=f"arn:aws:sns:{REGION}:{ACCOUNT_ID}:{SNS_TOPIC}",
        Message=message,
        Subject=subject
    )

    return {
        "statusCode": 200,
        "body": "Check complete. Notification sent."
    }

   
