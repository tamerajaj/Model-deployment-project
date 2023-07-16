import pytest
from src.helper_functions import get_data, split_data
import pandas as pd


def test_get_data():
    df = get_data()
    assert type(df) == pd.DataFrame
