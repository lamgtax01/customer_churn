import os
import joblib
import pandas as pd
from io import StringIO

def model_fn(model_dir):
    """Load the model from the directory."""
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

def input_fn(input_data, content_type):
    """Preprocess the input data."""
    if content_type == "text/csv":
        # Load CSV input data into a Pandas DataFrame
        data = pd.read_csv(StringIO(input_data), header=None)
        return data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Perform predictions."""
    predictions = model.predict(input_data)
    return predictions

def output_fn(predictions, accept):
    """Format the output predictions."""
    if accept == "text/csv":
        return "\n".join(map(str, predictions)), "text/csv"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
