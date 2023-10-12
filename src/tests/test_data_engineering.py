import pytest
from datetime import datetime

from main.data_preprocessing.data_engineering import Preprocess

import numpy as np
import pandas as pd


def test_get_dates_from_day_codes():
    preprocessor = _create_preprocessing_object()
    dates = preprocessor.get_dates_from_day_codes(day_codes=["d_1", "d_3"], 
                                            reference_date=datetime(2011,1,29), 
                                            direction="lead")
    assert all(dates == np.array([datetime(2011, 1, 29, 0, 0), datetime(2011, 1, 31, 0, 0)]))


def test_add_leads_and_lags_features():
    preprocessor = _create_preprocessing_object()
    sample_data = pd.DataFrame({"cat": ["cat1"]*5 + ["cat2"]*5,
                           "values": np.arange(1,11)})
    results = preprocessor.add_leads_and_lags_features(sample_data, 
                                                    column_name="values", 
                                                    group_by=["cat"], lags=[1, 2], 
                                                    leads=[2])

    assert np.isin(["values_lag_1", "values_lag_2", "target_2"], results.columns).all()
    assert ((results["values_lag_1"].values == \
            np.array([np.nan,  1.,  2.,  3.,  4., np.nan,  6.,  7.,  8.,  9.])) \
                                                | (np.isnan(results["values_lag_1"]))).all()
    assert ((results["values_lag_2"].values == \
            np.array([np.nan, np.nan,  1.,  2.,  3., np.nan, np.nan,  6.,  7.,  8.])) \
                                                | (np.isnan(results["values_lag_2"]))).all()
    assert ((results["target_2"].values == \
            np.array([ 2.,  3.,  4.,  5., np.nan,  7.,  8.,  9., 10., np.nan])) \
                                                | (np.isnan(results["target_2"]))).all()

def _create_preprocessing_object():
    preprocessing_parameters = {"raw_data_name": "sales_train_evaluation",
                                "processed_file_name": f"m5_processed_train",
                                "mode": "train"}
    return Preprocess(parameters=preprocessing_parameters)


