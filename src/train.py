# import json
import os
import sys

# import click
import mlflow

# import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error  # precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from helper_functions import get_data, preprocess

# from random_forrest_regressor import RandomForest

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


def train():
    """Train the model."""
    dataset = get_data()
    df_processed = preprocess(dataset)

    y = df_processed["trip_duration_minutes"]
    X = df_processed.drop(columns=["trip_duration_minutes"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    X_train = X_train.to_dict(orient="records")
    X_test = X_test.to_dict(orient="records")
    features = ["PULocationID", "DOLocationID", "trip_distance"]
    target = "duration"

    with mlflow.start_run():
        load_dotenv()
        year = os.getenv("YEAR")
        month = int(os.getenv("MONTH"))
        color = os.getenv("COLOR")
        tags = {
            "model": "linear regression pipeline",
            "developer": "<your name>",
            "dataset": f"{color}-taxi",
            "year": year,
            "month": month,
            "features": features,
            "target": target,
        }
        mlflow.set_tags(tags)
        pipeline = make_pipeline(
            DictVectorizer(),
            RandomForestRegressor(verbose=2, n_jobs=-1, n_estimators=10),
        )
        print("Training the model...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(pipeline, "model")


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
