import os

import mlflow

# import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

from helper_functions import get_data

_ = Pipeline
# get data
df = get_data()
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
# Set up the connection to MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Setup the MLflow experiment
mlflow.set_experiment("green-taxi-trip-duration")

features = ["PULocationID", "DOLocationID", "trip_distance"]
target = "duration"


# calculate the trip duration in minutes and drop trips that are less than 1 minute and more than 2 hours
def calculate_trip_duration_in_minutes(df):
    df["trip_duration_minutes"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["trip_duration_minutes"] >= 1) & (df["trip_duration_minutes"] <= 60)]
    return df


def preprocess(df):
    df = df.copy()
    df = calculate_trip_duration_in_minutes(df)
    categorical_features = ["PULocationID", "DOLocationID"]
    df[categorical_features] = df[categorical_features].astype(str)
    df["trip_route"] = df["PULocationID"] + "_" + df["DOLocationID"]
    df = df[["trip_route", "trip_distance", "trip_duration_minutes"]]
    return df


df_processed = preprocess(df)

y = df_processed["trip_duration_minutes"]
X = df_processed.drop(columns=["trip_duration_minutes"])


dv = DictVectorizer()


SA_KEY = os.getenv("GOOGLE_SA_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_KEY

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

X_train = X_train.to_dict(orient="records")
X_test = X_test.to_dict(orient="records")

with mlflow.start_run():
    tags = {
        "model": "linear regression pipeline",
        "developer": "<your name>",
        # "dataset": f"{color}-taxi",
        # "year": year,
        # "month": month,
        "features": features,
        "target": target,
    }
    mlflow.set_tags(tags)
    pipeline = make_pipeline(DictVectorizer(), LinearRegression())
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(pipeline, "model")
