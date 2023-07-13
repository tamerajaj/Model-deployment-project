import os
import sys

# import numpy as np
import optuna
import pandas as pd

# import skops.io as sio
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error  # , precision_recall_fscore_support

# from sklearn.model_selection import train_test_split

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import Normalizer, StandardScaler

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


def calculate_trip_duration_in_minutes(df):
    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["trip_duration_minutes"] >= 1) & (df["trip_duration_minutes"] <= 60)]
    return df


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


# def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
#     return train_test_split(X, y, test_size=test_size)


def optimize_hyperparameters(self, X_train, y_train, X_test, y_test):
    """
    Perform hyperparameter optimization with Optuna.
    """

    def objective(trial):
        # Define the search space for hyperparameters
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        max_depth = trial.suggest_int("max_depth", 2, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        # Create a decision tree classifier with the hyperparameters
        # TODO: change parameters
        dec_tree = RandomForestRegressor(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

        # Train the classifier
        dec_tree.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = dec_tree.predict(X_test)

        # Calculate the accuracy
        accuracy = mean_squared_error(y_test, y_pred)

        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # Print the best hyperparameters and the corresponding accuracy
    best_trial = study.best_trial
    print("Best hyperparameters:", best_trial.params)
    print("Best accuracy:", best_trial.value)
    return best_trial.params, study


def train_model_with_best_hyperparameters(model, X_train, y_train, best_params):
    """
    Train a decision tree model with the best hyperparameters.
    """
    model = model(**best_params)
    model.fit(X_train, y_train)
    return model
