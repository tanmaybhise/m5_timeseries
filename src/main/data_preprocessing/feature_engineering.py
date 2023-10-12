import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append("src")
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn import preprocessing
import pickle
from main.utils import utils
from main.utils.logger import logger as logging
from main.config import Config
from tqdm import tqdm

class FeatureEngineering():
    def __init__(self, parameters):
        self.bronze_data_path = Config.bronze_data_path
        self.processed_file_name = parameters["processed_file_name"]
        self.silver_data_path = Config.silver_data_path
        self.features_data_name = parameters["features_data_name"]
        self.feature_engineering_artifacts_path = parameters["feature_engineering_artifacts_path"]
        self.mode = parameters["mode"]

    def main(self):
        extract_df = self.extract()
        transform_df = self.transform(extract_df)
        self.load(transform_df)
    
    def extract(self):
        logging.info(f"Extract process started at {datetime.now()}")
        dataframe = pd.read_parquet(f"{self.bronze_data_path}/{self.processed_file_name}.parquet")
        logging.info(f"Extract process finished at {datetime.now()}")
        return dataframe

    def transform(self, dataframe):
        logging.info(f"Transform process started at {datetime.now()}")

        targets = [col for col in dataframe.columns if (col.find("target")!=-1)]
        lags = [col for col in dataframe.columns if (col.find("lag")!=-1)]
        other = ['id', 'prediction_start_date', 'wm_yr_wk', 
                'wday', 'month', 'year']
        snaps = [f"snap_{state}" for state in dataframe["state_id"].unique()]
        categorical_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'weekday']
        stats_encoded_features = []
        for cat in categorical_features:
            stats_encoded_features+=[f"{cat}_mean_demand", f"{cat}_max_demand", f"{cat}_std_demand"]
        boolean_features = ['event_name_1','event_name_2']

        dataframe["generated_id"] = dataframe["item_id"]+"_"+dataframe["store_id"]
        processed_data = self.convert_string_to_boolean(dataframe, boolean_features)
        
        processed_data = utils.create_stats_encoding(data=processed_data, 
                                                     features=categorical_features, 
                                                     mode=self.mode)
        selected_features = other+boolean_features+stats_encoded_features+snaps+lags+targets
        processed_data = processed_data[selected_features]
        
        numeric_columns = processed_data.columns[~processed_data.columns.isin(["prediction_start_date", "id"])]
        processed_data.loc[:, numeric_columns]=processed_data[numeric_columns].apply(pd.to_numeric)
        processed_data.loc[:, "prediction_start_date"] = pd.to_datetime(processed_data["prediction_start_date"])

        logging.info(f"Transform process finished at {datetime.now()}")
        return processed_data

    def load(self, data):
        logging.info(f"Load process started at {datetime.now()}")
        utils.make_dir_if_not_exist(self.silver_data_path)
        data.to_parquet(f"{self.silver_data_path}/{self.features_data_name}.parquet")
        logging.info(f"Processed data stored at {self.silver_data_path}/{self.features_data_name}.parquet")
        logging.info(f"Load process finished at {datetime.now()}")

    @staticmethod
    def convert_string_to_boolean(dataframe, boolean_features):
        dataframe.loc[:, boolean_features] = dataframe.loc[:, boolean_features].astype(bool).astype(int)
        return dataframe
    
    def create_categorical_label_encoders(self, dataframe, categorical_features):
        artifact = "label_encoder"
        for feature in categorical_features:
            utils.make_dir_if_not_exist(f"{self.feature_engineering_artifacts_path}/{artifact}")
            LE = preprocessing.LabelEncoder().fit(dataframe[feature])
            pickle.dump(LE, open(f'{self.feature_engineering_artifacts_path}/{artifact}/{feature}_{artifact}.pkl', 'wb'))

    def encode_categories(self, dataframe, categorical_features, encoder_type="label_encoder"):
        artifact=encoder_type
        for feature in categorical_features:
            decoder = pickle.load(open(f'{self.feature_engineering_artifacts_path}/{artifact}/{feature}_{artifact}.pkl', 'rb'))
            dataframe.loc[:, [feature]] = decoder.transform(dataframe[feature])

        return dataframe
    
if __name__=="__main__":
    feature_engineering_parameters = {
                            "processed_file_name": "m5_processed",
                            "features_data_name": "features",
                            "feature_engineering_artifacts_path": "src/data/feature_engineering_artifacts",
                            "mode": "train"}
    feature_engineering = FeatureEngineering(parameters=feature_engineering_parameters)
    feature_engineering.main()
