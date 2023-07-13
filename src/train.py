# import json
import os
import sys

# import click
import mlflow
import optuna

# import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error  # precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from helper_functions import get_data, preprocess

# from sklearn.pipeline import make_pipeline


# from random_forrest_regressor import RandomForest

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

# TODO: make logging work for all runs of optuna


def train():
    """Train the model."""
    dataset = get_data()
    # df_processed = preprocess(dataset)

    y = dataset["trip_duration_minutes"].values
    X = dataset.drop(columns=["trip_duration_minutes"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    dv = DictVectorizer()
    X_train, dv = preprocess(X_train, dv, fit_dv=True)
    X_test, _ = preprocess(X_test, dv, fit_dv=False)
    features = ["PULocationID", "DOLocationID", "trip_distance"]
    target = "duration"

    def objective(trial):
        # Define the search space for hyperparameters
        criterion = trial.suggest_categorical("criterion", ["squared_error"])
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
        n_estimators = trial.suggest_int("n_estimators", 10, 50)
        # Create a random forest regressor with the hyperparameters
        rf_regressor = RandomForestRegressor(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_estimators=n_estimators,
            n_jobs=-1,
        )

        # Train the regressor
        rf_regressor.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = rf_regressor.predict(X_test)

        # Calculate the root mean squared error (RMSE)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        return rmse

    # MLflow tracking
    with mlflow.start_run():
        load_dotenv()
        year = os.getenv("YEAR")
        month = int(os.getenv("MONTH"))
        color = os.getenv("COLOR")
        tags = {
            "model": "random forest regressor",
            "dataset": f"{color}-taxi",
            "year": year,
            "month": month,
            "features": features,
            "target": target,
        }
        mlflow.set_tags(tags)

        # Perform hyperparameter optimization with Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)

        # Get the best trial and parameters
        best_trial = study.best_trial
        best_params = best_trial.params

        # Log the Optuna search results to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("best_rmse", best_trial.value)

        # Create a random forest regressor with the best hyperparameters
        rf_regressor = RandomForestRegressor(
            criterion=best_params["criterion"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            n_estimators=best_params["n_estimators"],
            n_jobs=-1,
        )

        print("Training the model...")
        rf_regressor.fit(X_train, y_train)

        y_pred = rf_regressor.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        # Log the final metrics to MLflow
        mlflow.log_metric("rmse", rmse)

        # Log the model to MLflow
        mlflow.sklearn.log_model(rf_regressor, "model")


#
# @cli.command()
# @click.option(
#     "--data",
#     "-d",
#     help="Path to the input data if you want to predict the model on own data",
#     default=None,
# )
# @click.option("--review", "-r", help="Review to predict the sentiment of", default=None)
# @click.option("--path", "-p", help="Path to the model", default="./models/model.bin")
# def predict(data, review, path):
#     """Predict the sentiment of a review."""
#     if data is None and review is None:
#         # Load the data
#         dataset = load_dataset("imdb", split="test")
#         df = pd.DataFrame(dataset)
#     if review is not None:
#         df = pd.DataFrame({"text": [review]})
#     if data is not None:
#         # Load the data
#         df = pd.read_csv(data)
#     # Load the model
#     model = SentimentClassifier()
#     model.load(path)
#
#     # Predict the sentiment
#     sentiment = model.predict(df["text"])
#     click.echo("The prediction is was successful")
#     df_pred = pd.DataFrame({"text": df["text"], "label": sentiment})
#     df_pred.to_csv("./data/predictions.csv", index=False)
#
#     if review is not None:
#         click.echo(f"Sentiment of the review: {sentiment[0]}")
#
#
# @cli.command()
# @click.option(
#     "--data",
#     "-d",
#     help="Path to the input data if you want to evaluate " "the model on own data",
#     default=None,
# )
# @click.option("--path", "-p", help="Path to the model", default="./models/model.bin")
# def evaluate(data, path):
#     """Evaluate model on test data."""
#     if data is None:
#         # Load the data
#         dataset = load_dataset("imdb", split="test")
#         df = pd.DataFrame(dataset)
#     else:
#         # Load the data
#         df = pd.read_csv(data)
#     model = SentimentClassifier()
#     model.load(path)
#     performance = model.score(df["text"], df["label"])
#     print(json.dumps(performance, indent=2))

if __name__ == "__main__":
    train()
