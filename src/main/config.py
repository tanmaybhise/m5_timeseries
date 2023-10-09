from modeling.models.lightgbm_model import LGBModel
from sklearn.linear_model import LinearRegression

class Config:
    raw_data_path = "src/data/raw"
    bronze_data_path = "src/data/bronze"
    silver_data_path = "src/data/silver"
    gold_data_path = "src/data/gold"
    trained_models_path = "src/data/trained_models"

    state_id = "WI"
    horizon = 28
    lookback_multiple = 2

    models_dict = {"lgb": LGBModel(),
                   "linear": LinearRegression()}
    
    datasets = [
                ("evaluation", "sales_train_evaluation"),
                ("validation", "sales_train_validation")
               ]

    @classmethod
    def get_config(cls, mode):

        if mode == "train":
            raw_data = "sales_train_evaluation"
        elif mode == "infer":
            raw_data = "sales_train_validation"

        preprocessing_parameters = {
                                    "raw_data_name": raw_data,
                                    "processed_file_name": f"m5_processed_{mode}",
                                    "mode": mode}
        
        feature_engineering_parameters = {
                                "feature_engineering_artifacts_path": "src/data/feature_engineering_artifacts",
                                "processed_file_name": f"m5_processed_{mode}",
                                "features_data_name": f"features_{mode}",
                                "mode": mode}
        
        model_parameters = {"models": "linear,lgb",
                            "features_data_name": f"features_{mode}",
                            "mode": mode,
                            "prediction_data_suffix": ""
                            }
        
        return preprocessing_parameters, feature_engineering_parameters, model_parameters