import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Paths
input_path = "/opt/ml/processing/input"
output_train_path = "/opt/ml/processing/train"
output_val_path = "/opt/ml/processing/validation"
output_test_path = "/opt/ml/processing/test"

# Read Parquet file
df = pd.read_parquet(f"{input_path}/ccdata.parquet")

# Basic cleaning
df = df.dropna(subset=["description", "category"])  # Ensure no missing target/feature

# Optional: you could engineer more features like time features here
df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
df["weekday"] = pd.to_datetime(df["datetime"]).dt.weekday

# We'll only use a small set of features for modeling
features = ["description", "amount", "merchant", "location", "hour", "weekday", "category"]
df = df[features]

# Encode label as int (category -> 0-N)
df["label"] = df["category"].astype("category").cat.codes
df = df.drop(columns=["category"])

# Count per class
label_counts = df["label"].value_counts()
min_class_count = label_counts.min()
num_classes = df["label"].nunique()

# Check for safe stratified splitting
stratify_ok = min_class_count >= 3 and len(df) >= 10

# Train/val/test split
# Train-test-validation split
if stratify_ok:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["label"])
else:
    print("⚠️ Too few samples per class for stratified split — falling back to random split.")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=True)


# Save to Parquet
os.makedirs(output_train_path, exist_ok=True)
os.makedirs(output_val_path, exist_ok=True)
os.makedirs(output_test_path, exist_ok=True)

train_df.to_parquet(f"{output_train_path}/train.parquet", index=False)
val_df.to_parquet(f"{output_val_path}/validation.parquet", index=False)
test_df.to_parquet(f"{output_test_path}/test.parquet", index=False)
