import unittest
import pandas as pd
from etl import load_raw_data, transform_input_df


class TestETL(unittest.TestCase):

    def test_load(self):
        df = load_raw_data()
        assert isinstance(df, pd.DataFrame)

    def test_transform(self):
        df = load_raw_data()
        df = transform_input_df(df)
        assert isinstance(df, pd.DataFrame)
