import os
import joblib
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import torch
import json
from sklearn.preprocessing import LabelEncoder

# Load training data
train_path = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
val_path = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")

train_df = pd.read_parquet(f"{train_path}/train.parquet")
val_df = pd.read_parquet(f"{val_path}/validation.parquet")

# Load tokenizer and encode text
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Encode categorical features
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["label"])
val_df["label"] = label_encoder.transform(val_df["label"])


def tokenize(example):
    return tokenizer(example["description"], padding="max_length", truncation=True)


train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(tokenize)
val_dataset = val_dataset.map(tokenize)

# Model setup
num_labels = len(np.unique(train_df["label"]))
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="/opt/ml/model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    logging_dir="./logs",
    logging_steps=10
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("/opt/ml/model")
tokenizer.save_pretrained("/opt/ml/model")
joblib.dump(label_encoder, "/opt/ml/model/label_encoder.joblib")

