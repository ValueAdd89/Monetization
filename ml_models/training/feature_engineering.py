import pandas as pd
import mlflow

def prepare_features(data):
    mlflow.log_param("input_rows", len(data))
    # Dummy feature logic
    return data
