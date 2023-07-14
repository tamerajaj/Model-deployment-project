import os
import uuid

import click
import mlflow
import pandas as pd


def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df["trip_duration_minutes"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.trip_duration_minutes = df.trip_duration_minutes.dt.total_seconds() / 60
    df = df[(df.trip_duration_minutes >= 1) & (df.trip_duration_minutes <= 60)]

    df["ride_id"] = generate_uuids(len(df))

    return df


def preprocess(df):
    df = df.copy()
    categorical_features = ["PULocationID", "DOLocationID"]
    df[categorical_features] = df[categorical_features].astype(str)

    df["trip_route"] = df["PULocationID"] + "_" + df["DOLocationID"]
    dicts = df[["trip_route", "trip_distance"]].to_dict(orient="records")

    return dicts


def load_model(run_id, MLFLOW_TRACKING_URI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logged_model = f"runs:/{run_id}/model"
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model


def save_results(df, y_pred, run_id, output_filename):
    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["lpep_pickup_datetime"] = df["lpep_pickup_datetime"]
    df_result["PULocationID"] = df["PULocationID"]
    df_result["DOLocationID"] = df["DOLocationID"]
    df_result["actual_duration"] = df["trip_duration_minutes"]
    df_result["predicted_duration"] = y_pred
    df_result["diff"] = df_result["actual_duration"] - df_result["predicted_duration"]
    df_result["model_version"] = run_id
    df_result.to_parquet(output_filename, index=False)


def apply_model(filename, run_id, output_filename, MLFLOW_TRACKING_URI):
    df = read_dataframe(filename)
    dicts = preprocess(df)

    loaded_model = load_model(run_id, MLFLOW_TRACKING_URI)
    y_pred = loaded_model.predict(dicts)

    save_results(df, y_pred, run_id, output_filename)


@click.command()
@click.option("--filename", help="Path to the input parquet file")
@click.option("--run_id", help="MLflow run ID")
@click.option("--output_filename", help="Path to the output parquet file")
@click.option("--MLFLOW_TRACKING_URI", help="MLflow tracking URI")
@click.option("--google_sa_key", help="Path to the Google SA key")
def run(filename, run_id, output_filename, MLFLOW_TRACKING_URI, google_sa_key):
    filename = filename
    output_filename = output_filename
    run_id = run_id
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_sa_key
    apply_model(filename, run_id, output_filename, MLFLOW_TRACKING_URI)


if __name__ == "__main__":
    run()
