import os
import sys

import numpy as np

# import optuna
import pandas as pd
import skops.io as sio
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
from sklearn.pipeline import Pipeline  # make_pipeline

# from sklearn.preprocessing import Normalizer, StandardScaler

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


class RandomForest:
    def __init__(self):
        # TODO: ad drop for time column

        self.model = None
        self.values = None

    # def preprocess(self, df: pd.DataFrame) -> None:
    #     self.preprocessor = Pipeline(
    #         [
    #             ("normalizer", Normalizer()),
    #             ("scaler", StandardScaler()),
    #         ]
    #     )
    # return self.preprocessor.fit_transform(df)  # TODO: fit preprocessor only to train data

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to the data."""
        self.values = y
        self.model = Pipeline(
            [
                ("classifier", RandomForestRegressor()),
            ]
        )
        self.model.fit(X, y)

    def predict(self, X: pd.Series) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)

    def score(self, X: pd.Series, y: pd.Series) -> dict:
        """Score the model on the test data."""
        y_pred = self.predict(X)

        # accuracy = mean_squared_error(y, y_pred)
        metrics = precision_recall_fscore_support(y, y_pred, average="weighted")
        performance = {
            "RMSE": round(mean_squared_error, 2),
            "precision": round(metrics[0], 2),
            "recall": round(metrics[1], 2),
            "f1": round(metrics[2], 2),
        }

        return performance

    def save(self, path: str) -> None:
        """Save the model to a file."""
        with open(f"{path}", "wb") as f_out:
            sio.dump(self.model, f_out)

    def load(self, path: str) -> None:
        """Load the model from a file."""
        # loading the model
        with open("./models/model.bin", "rb") as _:
            self.model = sio.load(path, trusted=True)
