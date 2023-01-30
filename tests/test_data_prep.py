import unittest

import pandas as pd

from src.data_prep import prepare_dataset


class MyTest(unittest.TestCase):
    def test_prepare_dataset(self):
        df_train, df_test = prepare_dataset(
            drop_columns=["airline_sentiment_confidence"],
            rename_columns={"airline_sentiment": "label"},
        )

        assert isinstance(df_train, pd.DataFrame)
        assert isinstance(df_test, pd.DataFrame)

        assert "airline_sentiment_confidence" not in df_test.columns
        assert "airline_sentiment_confidence" not in df_train.columns

        assert "airline_sentiment" not in df_test.columns
        assert "airline_sentiment" not in df_train.columns

        assert "label" in df_test.columns
        assert "label" in df_train.columns
