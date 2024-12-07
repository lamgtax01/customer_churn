# Scikit-Learn SageMaker Pipeline Template

This folder provides a comprehensive template for building, training, evaluating, and deploying Machine Learning pipelines using **Scikit-Learn** on AWS SageMaker. It includes all necessary artifacts, code, configuration files, and visualizations to streamline your ML workflow.

## Table of Contents

- [Folder Structure](#Folder-Structure)
- [Getting Started](#getting-started)
  - [1. Customize Configuration](#1.-Customize-Configuration)
  - [2. Data Preprocessing](#2.-Data-Preprocessing)
  - [3. Train the Model](#3.-Train-the-Model)
  - [4. Evaluate the Model](#4.-Evaluate-the-Model)
  - [5. Deploy and Inference](#5.-Deploy-and-Inference)
- [Artifacts](#Artifacts)
- [Notebook](#Notebook)
- [Usage Notes](#Usage-Notes)
- [Visualization](#Visualization)


## Folder Structure

```plaintext
Scikit-learn/
├── Artifacts/           # Contains the trained model artifact
│   └── model.tar.gz
├── Code/                # Contains the scripts for various stages of the ML pipeline
│   ├── preprocessing.py           # Data preprocessing script
│   ├── preprocessing-pyspark.py   # Alternative preprocessing with PySpark
│   ├── train.py                   # Training script
│   ├── evaluation.py              # Model evaluation script
│   ├── inference.py               # Inference script for predictions
├── Evaluation/          # Stores the evaluation report
│   └── evaluation.json
├── Config/              # Contains configuration files for the pipeline
│   └── config.json
├── images/              # Visual assets for the pipeline
│   ├── pipeline_graph.jpeg        # Pipeline workflow graph
│   ├── lineage_graph.jpeg         # Lineage graph of artifacts
│   └── model_version.jpeg         # Model version tracking image
├── scik-Learn-pipeline-template-v0x.ipyb  # Jupyter notebook for building and running the pipeline
```

## Getting Started
Follow these steps to use the Scikit-learn SageMaker Pipeline Template:

### 1. Install Dependencies
Ensure you have the required dependencies installed. Use the requirements.txt file or manually install the key libraries:

```bash
pip install sagemaker scikit-learn boto3 pandas matplotlib
```

### 1. Customize Configuration
Update the **`Config/config.json`** file with your specific project and AWS environment details:

- S3 bucket names for input and output data.
- Instance types for training and inference.
- Model hyperparameters.

### 2. Data Preprocessing
Use one of the preprocessing scripts from the *`Code/`* folder:

- **`preprocessing.py`**: Standard Scikit-learn preprocessing.
- **`preprocessing-pyspark.py`**: For handling large-scale datasets using PySpark.

### 3. Train the Model
- Train your model using *`train.py`*:
- This script outputs a trained model artifact *`model.tar.gz`* in the *`Artifacts/`* folder.

### 4. Evaluate the Model
Evaluate the model using the evaluation.py script. The results will be saved in *`Evaluation/evaluation.json`*

### 5. Deploy and Infer
Deploy your trained model on SageMaker and use *`inference.py`* to make predictions:


## Artifacts
- **`Model Artifact`**: The *`model.tar.gz`* file in the *`Artifacts/`* folder contains the serialized trained model.
- **`Evaluation Report`**: The *`Evaluation/evaluation.json`* file summarizes key metrics like accuracy, precision, recall, and F1 score.


## Notebook
The notebook **`skl-sagemaker-pipeline-template-vxx.ipynb`** is a step-by-step guide for:

- Building and running the pipeline.
- Visualize results.


## Usage Notes
- The scripts and notebook are designed to be modular, so you can easily adapt them to your specific project.


## Visualization
The images/ folder contains visualizations for better understanding of the pipeline:
- pipeline_graph.jpeg: Workflow diagram of the SageMaker pipeline.
- lineage_graph.jpeg: Lineage graph showing data and model dependencies.
- model_version.jpeg: Snapshot of the model version registered in SageMaker.



