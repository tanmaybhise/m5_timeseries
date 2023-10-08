import pytest
from datetime import datetime

from main.data_preprocessing.data_engineering import Preprocess

import numpy as np


def test_get_dates_from_day_codes():
    preprocessor = _create_preprocessing_object()
    dates = preprocessor.get_dates_from_day_codes(day_codes=["d_1", "d_3"], 
                                            reference_date=datetime(2011,1,29), 
                                            direction="lead")
    assert all(dates == np.array([datetime(2011, 1, 29, 0, 0), datetime(2011, 1, 31, 0, 0)]))

def test_split_lags_and_targets():
    preprocessor = _create_preprocessing_object()
    day_codes = [f"{n+1}" for n in range(10)]
    lags_and_targets = preprocessor.split_lags_and_targets(day_codes, horizon=preprocessor.horizon, 
                                                           lookback_multiple=preprocessor.lookback_multiple)

    assert lags_and_targets[0].lags == ['5', '6', '7', '8'] and \
    lags_and_targets[0].targets == ['9', '10']
    assert lags_and_targets[1].lags == ['3', '4', '5', '6'] and \
        lags_and_targets[1].targets == ['7', '8']
    assert lags_and_targets[2].lags == ['1', '2', '3', '4'] and \
        lags_and_targets[2].targets == ['5', '6']

def _create_preprocessing_object():
    preprocessing_parameters = {"state_id":"WI",
                                "horizon": 2,
                                "lookback_multiple": 2,
                                "raw_data_path": "src/data/raw",
                                "bronze_data_path": "src/data/bronze",
                                "processed_file_name": "m5_processed"}
    return Preprocess(preprocessing_parameters)


