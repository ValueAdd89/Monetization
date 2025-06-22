import mlflow

def train_ltv_model():
    with mlflow.start_run():
        mlflow.log_param("model_type", "regression")
        print("Training LTV model with MLflow...")
