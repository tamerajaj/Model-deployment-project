import os

import mlflow
import pandas as pd
from dotenv import load_dotenv


def prepare_features(ride):
    features = {}
    features["trip_route"] = f"{ride.PULocationID}_{ride.DOLocationID}"
    features["trip_distance"] = ride.trip_distance
    return features


def load_model(model_name):
    stage = "Production"
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict(model_name, data):
    load_dotenv()
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_input = prepare_features(data)
    model = load_model(model_name)
    prediction = model.predict(model_input)
    return float(prediction[0])


def store_in_bq(data):
    load_dotenv()
    TABLE_NAME = " w3_project_yellow_taxi_ml_api.yellow_taxi_ml_api_predictions"  # os.getenv("TABLE_NAME")
    df = pd.DataFrame([pred.dict() for pred in data])
    df.to_gbq(destination_table=TABLE_NAME, if_exists="append")
