import pytest

import pandas as pd

from main.data_preprocessing.feature_engineering import FeatureEngineering

def test_convert_string_to_boolean():
    fe = _get_feature_engineering_object()
    sample_data = pd.DataFrame({
                               "f1": [None, None, "String", None],
                               "f2": ["String", None, "String", None],
                               "f3": [1.,2,3,4]
                               })
    results = fe.convert_string_to_boolean(sample_data, ["f1", "f2"])
    assert all(results["f1"] == [0,0,1,0])
    assert all(results["f2"] == [1,0,1,0])
    assert all(results["f3"] == [1.,2.,3.,4.])

def _get_feature_engineering_object():
    feature_engineering_parameters = {
                            "bronze_data_path": "src/data/bronze",
                            "processed_file_name": "m5_processed",
                            "silver_data_path": "src/data/silver",
                            "features_data_name": "features",
                            "feature_engineering_artifacts_path": "src/data/feature_engineering_artifacts"}
    fe = FeatureEngineering(parameters=feature_engineering_parameters)
    return fe