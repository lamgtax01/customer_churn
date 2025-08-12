import os
import logging
import joblib
import xgboost
import numpy as np

# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def model_fn(model_dir):
    """Load the model once when the container starts."""
    logger.info("Loading model from: %s", model_dir)
    model_path = os.path.join(model_dir, "xgboost_model.joblib")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, content_type):
    """Parse text/csv into xgboost.DMatrix (features only, no header/label)."""
    if (content_type or "").split(";")[0].strip().lower() != "text/csv":
        raise ValueError(f"Unsupported content type: {content_type}")

    if isinstance(request_body, (bytes, bytearray)):
        body = request_body.decode("utf-8")
    else:
        body = request_body

    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")  # comma-separated
        rows.append([float(x) for x in parts])

    if not rows:
        raise ValueError("Empty CSV payload.")

    arr = np.asarray(rows, dtype=np.float32)
    return xgboost.DMatrix(arr)

def predict_fn(input_data, model):
    """Return probabilities for binary:logistic."""
    logger.info("Performing inference...")
    probs = model.predict(input_data)

    # Normalize to 1-D (handles shapes like (N,1) or (N,2))
    if isinstance(probs, np.ndarray):
        if probs.ndim == 2 and probs.shape[1] == 2:
            probs = probs[:, 1]
        elif probs.ndim == 2 and probs.shape[1] == 1:
            probs = probs[:, 0]
    return probs  # 1-D array/list of probabilities

def output_fn(prediction_output, accept):
    """Return text/csv (one probability per line)."""
    if accept and (accept.split(";")[0].strip().lower() != "text/csv"):
        raise ValueError(f"Unsupported Accept type: {accept}")

    probs = prediction_output.tolist() if hasattr(prediction_output, "tolist") else prediction_output
    lines = [str(float(p)) for p in probs]
    return "\n".join(lines), "text/csv"
