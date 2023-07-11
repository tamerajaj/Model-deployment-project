import json
import os
import sys

import click
import pandas as pd
from helper_functions import get_data, split_data
from random_forrest_regressor import RandomForest
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--data",
    "-d",
    help="Path to the input data if you want to train" " the model on own data",
    default=None,
)
@click.option("--path", "-p", help="Path to the save the model to", default=None)
def train(data, path):
    """Train the model."""
    dataset = get_data()

    X = dataset.drop("total_amount", axis=1)
    X.drop(["tpep_pickup_datetime","tpep_dropoff_datetime"], axis=1, inplace=True)
    y = dataset["total_amount"]
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    model = RandomForest()

    X_train_preprocessed = model.preprocessor.fit_transform(X_train)
    X_test_preprocessed = model.preprocessor.transform(X_test)

    # Train the model
    model.fit(X_train, y_train)
    click.echo("Trained the model successfully")
    if path is not None:
        # Save the model
        model.save(path)
        click.echo(f"Saved the model to {path}")

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
    cli()
