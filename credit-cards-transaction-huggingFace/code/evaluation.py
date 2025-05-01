import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 🔧 Paths
model_dir = "/opt/ml/processing/model"
test_path = "/opt/ml/processing/test/test.parquet"
output_dir = "/opt/ml/processing/evaluation"
os.makedirs(output_dir, exist_ok=True)

# 🧪 Load test data
df = pd.read_parquet(test_path)
df = df.dropna(subset=["description", "label"])

# Encode labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Load model & tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.eval()

# Tokenize
dataset = Dataset.from_pandas(df)
dataset = dataset.map(lambda x: tokenizer(x["description"], padding="max_length", truncation=True), batched=True)
labels = df["label"].values
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Predict
pred_logits = []
with torch.no_grad():
    for batch in torch.utils.data.DataLoader(dataset, batch_size=32):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        pred_logits.append(outputs.logits.cpu().numpy())

pred_logits = np.concatenate(pred_logits, axis=0)
preds = np.argmax(pred_logits, axis=1)

# 🧮 Metrics
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, average="weighted", zero_division=0)
recall = recall_score(labels, preds, average="weighted", zero_division=0)
f1 = f1_score(labels, preds, average="weighted", zero_division=0)

# AUC only if binary
try:
    auc = roc_auc_score(labels, pred_logits[:, 1]) if pred_logits.shape[1] == 2 else float("nan")
except Exception:
    auc = float("nan")

# ✅ AWS-formatted metrics
report_dict = {
    "binary_classification_metrics": {
        "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
        "precision": {"value": precision, "standard_deviation": "NaN"},
        "recall": {"value": recall, "standard_deviation": "NaN"},
        "f1": {"value": f1, "standard_deviation": "NaN"},
        "auc": {"value": auc, "standard_deviation": "NaN"},
    }
}

# Save AWS-style metrics
with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
    json.dump(report_dict, f)

print("✅ Evaluation metrics saved:", report_dict)

# 📊 Save ROC & PR curves
def save_plot(y_true, y_score, title, filename):
    plt.figure()
    if title == "ROC":
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
    elif title == "Precision-Recall":
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, label="PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{filename}.jpg"))
    plt.close()

# Only for binary classification
if pred_logits.shape[1] == 2:
    scores = pred_logits[:, 1]
    save_plot(labels, scores, "ROC", "roc")
    save_plot(labels, scores, "Precision-Recall", "pr")
else:
    print("⚠️ ROC/PR curves skipped — not binary classification.")
