import os

import mlflow
import pandas as pd


def prepare_features(ride):
    features = {}
    features["trip_route"] = f"{ride.PULocationID}_{ride.DOLocationID}"
    features["trip_distance"] = ride.trip_distance
    return features


def load_model(model_name):
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    print(MLFLOW_TRACKING_URI)

    stage = "production"
    model_uri = f"models:/{model_name}/{stage}"
    print(model_uri)
    model = mlflow.pyfunc.load_model(model_uri)

    return model


def predict(model_name, data):
    print(model_name)
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    print(MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = load_model(model_name)
    model_input = prepare_features(data)

    print(data)
    print(model_input)
    prediction = model.predict(model_input)
    return float(prediction[0])


def store_in_bq(data):
    TABLE_NAME = "w3_project_yellow_taxi_ml_api.yellow_taxi_ml_api_predictions"  # os.getenv("TABLE_NAME")
    df = pd.DataFrame([pred.dict() for pred in data])
    df.to_gbq(destination_table=TABLE_NAME, if_exists="append")


if __name__ == "__main__":
    load_model("random-forest-fare-model")
