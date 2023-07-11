from src.pipeline.custom_transformer_seattle import (TempMinTransformer,
                                                     WeatherColumnTransformer,
                                                     FloatColumnTransformer,
                                                     DateColumnTransformer)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler,
                                   FunctionTransformer,
                                   OneHotEncoder)
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PreprocessingYellowTaxi():
    def __init__(self):
        self.numerical_features = []
        self.categorical_features = []

        self.data_cleaning_pipeline = Pipeline([
            ('passenger_count', PassengerCountTransformer()),
            ('trip_duration', TripDurationTransformer()),
        ]
        )


class PassengerCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X[(X['passenger_count'] > 0) & (X['passenger_count'] <= 6)]
        return X


class TripDurationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['trip_duration'] = X['tpep_dropoff_datetime'] - X[
            'tpep_pickup_datetime']
        X = X[(X['trip_duration'] > 0) & (X['trip_duration'] <= 3600)]
        return X
