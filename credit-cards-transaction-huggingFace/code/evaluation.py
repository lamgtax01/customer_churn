# evaluate.py

import pandas as pd
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

# Load test data
test_path = "/opt/ml/processing/test/test.parquet"
df = pd.read_parquet(test_path)

# Encode labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Load model
model_dir = "/opt/ml/processing/model"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Tokenize
dataset = Dataset.from_pandas(df)
def preprocess(example):
    return tokenizer(example["description"], padding="max_length", truncation=True)
dataset = dataset.map(preprocess, batched=True)

# Predict
model.eval()
inputs = dataset.remove_columns(["description", "label"]).with_format("torch")
labels = dataset["label"]
preds = []

with torch.no_grad():
    for batch in torch.utils.data.DataLoader(inputs, batch_size=32):
        outputs = model(**batch)
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        preds.extend(batch_preds)

# Classification Report
report = classification_report(labels, preds, output_dict=True)
accuracy = report["accuracy"]
precision = report["weighted avg"]["precision"]
recall = report["weighted avg"]["recall"]
f1 = report["weighted avg"]["f1-score"]

metrics = {
    "metrics": {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
}

os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)
with open("/opt/ml/processing/evaluation/metrics.json", "w") as f:
    json.dump(metrics, f)

# Curves
def save_plot(y_true, y_pred, title, filename):
    plt.figure()
    if title == "ROC":
        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
    elif title == "Precision-Recall":
        precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
        plt.plot(recall, precision, label="PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.savefig(f"/opt/ml/processing/evaluation/{filename}.jpg")

# For multiclass AUC, this is simplified; in real cases use OneVsRest
save_plot(labels, preds, "ROC", "roc")
save_plot(labels, preds, "Precision-Recall", "pr")


