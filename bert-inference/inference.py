# inference.py

import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List
import json

# Load model and tokenizer
def model_fn(model_dir):
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return {'model': model, 'tokenizer': tokenizer}

# Input is CSV-style text: one transaction per line
def input_fn(input_data, content_type):
    if content_type == 'text/csv':
        lines = input_data.strip().split('\n')
        return lines
    raise ValueError(f"Unsupported content type: {content_type}")

# Preprocess lines into token IDs
def predict_fn(input_data, model_artifacts):
    model = model_artifacts['model']
    tokenizer = model_artifacts['tokenizer']
    model.eval()

    inputs = tokenizer(input_data, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, axis=1)

    return preds.tolist()

# Output raw predictions
def output_fn(prediction, accept):
    if accept == 'text/csv':
        return '\n'.join(str(p) for p in prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
