class Parameters:
    preprocessing_parameters = {"state_id":"WI",
                                "horizon": 28,
                                "lookback_multiple": 2,
                                "raw_data_path": "src/data/raw",
                                "bronze_data_path": "src/data/bronze",
                                "processed_file_name": "m5_processed"}
    
    feature_engineering_parameters = {
                            "bronze_data_path": "src/data/bronze",
                            "processed_file_name": "m5_processed",
                            "silver_data_path": "src/data/silver",
                            "features_data_name": "features",
                            "feature_engineering_artifacts_path": "src/data/feature_engineering_artifacts"}
    
