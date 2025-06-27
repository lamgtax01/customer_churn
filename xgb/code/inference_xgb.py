import json
import os
import logging
import joblib
import xgboost
import numpy as np

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def model_fn(model_dir):
    """
    Load the model from the model_dir. This is called once when the container is started.
    """
    logger.info("Loading model from: %s", model_dir)
    model_path = os.path.join(model_dir, "xgboost_model.joblib")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, content_type):
    """
    Convert the input request JSON to a DMatrix for XGBoost.
    Expects application/json with a list of feature vectors.
    """
    logger.info(f"Received content type: {content_type}")

    if content_type == 'application/json':
        # Expect input like: {"instances": [[f1, f2, ...], [f1, f2, ...]]}
        input_data = json.loads(request_body)

        if isinstance(input_data, dict) and "instances" in input_data:
            data = input_data["instances"]
        else:
            data = input_data  # Allow raw list as fallback

        dmatrix = xgboost.DMatrix(np.array(data))
        return dmatrix
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Run the prediction on the input DMatrix using the loaded model.
    """
    logger.info("Performing inference...")
    prediction_probs = model.predict(input_data)
    predictions = prediction_probs.round().astype(int)  # 0 or 1
    return {
        "predictions": predictions.tolist(),
        "probabilities": prediction_probs.tolist()
    }


def output_fn(prediction_output, accept):
    """
    Serialize the prediction response.
    """
    logger.info(f"Preparing output with accept: {accept}")

    if accept == "application/json":
        return json.dumps(prediction_output), "application/json"
    else:
        raise ValueError(f"Unsupported Accept type: {accept}")
