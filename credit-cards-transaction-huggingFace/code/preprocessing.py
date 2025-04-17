# preprocessing.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Paths
input_path = "/opt/ml/processing/input"
output_train_path = "/opt/ml/processing/train"
output_val_path = "/opt/ml/processing/validation"
output_test_path = "/opt/ml/processing/test"

# Read Parquet file
df = pd.read_parquet(f"{input_path}/transactions.parquet")

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

# Train/val/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["label"])

# Save to Parquet
os.makedirs(output_train_path, exist_ok=True)
os.makedirs(output_val_path, exist_ok=True)
os.makedirs(output_test_path, exist_ok=True)

train_df.to_parquet(f"{output_train_path}/train.parquet", index=False)
val_df.to_parquet(f"{output_val_path}/validation.parquet", index=False)
test_df.to_parquet(f"{output_test_path}/test.parquet", index=False)
