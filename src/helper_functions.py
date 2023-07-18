import os
import sys

import pandas as pd
from dotenv import load_dotenv

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


# TODO: add option to change year, month, and color.
def get_data() -> pd.DataFrame:
    """Load the data from a file."""
    load_dotenv()
    year = os.getenv("YEAR")
    month = int(os.getenv("MONTH"))
    color = os.getenv("COLOR")
    if not os.path.exists(f"./data/" f"{color}_tripdata_{year}-{month:02d}.parquet"):
        print("Loading data from URL...")
        os.system(
            f"wget -P ./data https://d37ci6vzurychx.cloudfront.net/trip-data/"
            f"{color}_tripdata_{year}-{month:02d}.parquet"
        )
        print("Successfully saved the data!")

    df = pd.read_parquet(f"./data/{color}_tripdata_{year}-{month:02d}.parquet")
    df = calculate_trip_duration_in_minutes(df)
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


def calculate_trip_duration_in_minutes(df: pd.DataFrame) -> pd.DataFrame:
    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["trip_duration_minutes"] >= 1) & (df["trip_duration_minutes"] <= 60)]
    return df


def preprocess(df: pd.DataFrame) -> list[dict]:
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")

    return dicts
