import pandas as pd
import pickle

def run_batch(input_path, model_path):
    data = pd.read_csv(input_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict(data.drop(columns=["user_id"]))
    return pd.DataFrame({"user_id": data["user_id"], "churn_pred": predictions})
