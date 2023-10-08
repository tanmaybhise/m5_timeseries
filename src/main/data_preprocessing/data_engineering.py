import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import zipfile
import os
from datetime import datetime, timedelta
from collections import namedtuple
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(funcName)s:%(name)s:%(message)s')
from main.utils import utils

class Preprocess():
    def __init__(self, parameters):
        """
         Initialize the instance. This is called by the __init__ method of the : class : ` Horizon ` class
         
         Args:
         	 parameters: Dictionary containing the parameters
        """
        self.state_id = parameters["state_id"].split(",")
        self.horizon = parameters["horizon"]
        self.lookback_multiple = parameters["lookback_multiple"]
        self.raw_data_path = parameters["raw_data_path"]
        self.bronze_data_path = parameters["bronze_data_path"]
        self.processed_file_name = parameters["processed_file_name"] 

    def main(self):
        extract_df = self.extract()
        transform_df = self.transform(extract_df)
        self.load(transform_df)

    def extract(self):
        logging.info(f"Extract process started at {datetime.now()}")
        utils.make_dir_if_not_exist(self.raw_data_path)
        if len(os.listdir(self.raw_data_path)) == 0:
            try:
                import kaggle
                kaggle.api.authenticate()
            except OSError as e:
                logging.error("Make sure you configure kaggle.json file. More info here: https://www.kaggle.com/docs/api")
                raise e
            kaggle.api.competition_download_files('m5-forecasting-accuracy', path=self.raw_data_path)
            with zipfile.ZipFile("src/data/raw/m5-forecasting-accuracy.zip", 'r') as zip_ref:
                zip_ref.extractall(self.raw_data_path)
            os.remove("src/data/raw/m5-forecasting-accuracy.zip")

        dataframe = pd.read_csv(self.raw_data_path+"/sales_train_evaluation.csv")

        logging.info(f"Extract process finished at {datetime.now()}")
        return dataframe

    def transform(self, dataframe):
        """
         Transforms data to be used in preprocessing. This is a method to be called by the transform_to_data method
         
         Returns: 
         	 a list of data
        """
        logging.info(f"Transform process started at {datetime.now()}")
        static_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        day_codes = np.array([f"d_{n}" for n in range(1,1942)])
        dataframe = dataframe[dataframe["state_id"].isin(self.state_id)]
        preprocessed_data = self.create_lags_and_target_columns(dataframe, static_columns, day_codes)
        calender_data = pd.read_csv(self.raw_data_path+"/calendar.csv")
        preprocessed_data = self.add_calender_features(preprocessed_data, calender_data)
        logging.info(f"Transform process finished at {datetime.now()}")
        return preprocessed_data
    
    def load(self, data):
        logging.info(f"Load process started at {datetime.now()}")
        utils.make_dir_if_not_exist(self.bronze_data_path)
        data.to_parquet(f"{self.bronze_data_path}/{self.processed_file_name}.parquet")
        logging.info(f"Processed data stored at {self.bronze_data_path}/{self.processed_file_name}.parquet")
        logging.info(f"Load process finished at {datetime.now()}")

    @staticmethod
    def get_dates_from_day_codes(day_codes, reference_date=datetime(2011,1,29), direction="lead"):
        """
         Get dates from day codes. This is a convenience function to get the date corresponding to a list of day codes
         
         Args:
         	 day_codes: a list of day codes
         	 reference_date: the date to use as reference default to (2011,1,29)
         	 direction: the direction of the day codes
         
         Returns: 
         	 a numpy array of
        """
        def _get_date_from_day_code(day_code, reference_date, direction):
            """
             Get date from day code. This is used to determine the date of a day in the calendar based on the day code.
             
             Args:
             	 day_code: The day code to be parsed.
             	 reference_date: The date that the day code is related to.
             	 direction: The direction of the day code. Either " lag " or " lead ".
             
             Returns: 
             	 The date of the day code in the reference date
            """
            prefix = day_code.split("_")[0]
            day_int = int(day_code.split("_")[-1])
            # If prefix is d then day_int - 1
            if prefix == "d":
                day_int=day_int-1
            # Return the date of the reference date
            if direction=="lag":
                return reference_date - timedelta(days=day_int)
            elif direction=="lead":
                return reference_date + timedelta(days=day_int)
        f = np.vectorize(_get_date_from_day_code)
        return f(day_codes, reference_date, direction)

    @staticmethod
    def split_lags_and_targets(day_codes, horizon, lookback_multiple):
        """
         Split lags and targets. This is a helper function for : func : ` get_lags_and_targets `.
         
         Args:
         	 day_codes: A list of day codes. Each entry is a numpy array of length ` ` len ( day_codes ) ` `.
         	 horizon: The number of days between lag and target.
         	 lookback_multiple: The number of lookbacks to be used for splitting.
         
         Returns: 
         	 A list of ` ` ( lags, targets ) ` ` tuples where ` ` lags ` ` is a numpy array of length ` ` len ( day_codes ) ` `
        """
        lags_and_targets_tuple = namedtuple("lags_and_targets_tuple", "lags targets")
        lags_and_targets = []
        lag_start_index = 0.1 #random small number
        n=0
        # Find the lag_start_index of the first lag in the day_codes.
        while abs(lag_start_index) < len(day_codes):
            target_start_index = -horizon*(n+1)
            target_end_index = [None if n==0 else -horizon*(n)][0]
            lag_start_index = -horizon*(n+1) - horizon*lookback_multiple
            # If lag_start_index is less than the number of day codes in the day codes list.
            if abs(lag_start_index) > len(day_codes):
                break
            lag_end_index = -horizon*(n+1)
            targets = day_codes[target_start_index:target_end_index]
            lags = day_codes[lag_start_index:lag_end_index]
            lags_and_targets.append(lags_and_targets_tuple(lags, targets))
            n+=1
        return lags_and_targets

    def create_lags_and_target_columns(self, dataframe, static_columns, day_codes):
        """
         Create lags and targets columns based on day codes. This is a helper function for : meth : ` create_all_lags_and_targets `
         
         Args:
         	 dataframe: Dataframe to be processed.
         	 static_columns: List of columns to be used for static prediction.
         	 day_codes: List of day codes to be used for splitting.
         
         Returns: 
         	 Processed dataframe
        """
        processed_df = pd.DataFrame()
        lags_and_targets = self.split_lags_and_targets(day_codes, horizon=self.horizon, lookback_multiple=self.lookback_multiple)
        # Returns a dataframe with the data for each batch of lags and targets.
        for batch in lags_and_targets:
            lags = batch.lags.tolist()
            targets = batch.targets.tolist()

            selected_columns = static_columns+lags+targets
            tmp_df = dataframe[selected_columns]
            tmp_df.loc[:, ["prediction_start_date"]] = self.get_dates_from_day_codes(targets[0])
            tmp_df.loc[:, ["prediction_start_date"]] = pd.to_datetime(tmp_df["prediction_start_date"]).dt.date

            column_mappings = dict()
            lags_mapping = {lag_column:f"lag_{n+1}" for lag_column,n in zip(lags, reversed(range(len(lags))))}
            column_mappings.update(lags_mapping)
            targets_mapping = {target_column:f"target_{n+1}" for target_column,n in zip(targets, range(len(targets)))}
            column_mappings.update(targets_mapping)

            tmp_df = tmp_df.rename(columns = column_mappings)
            processed_df = pd.concat([processed_df, tmp_df], axis=0)
        
        processed_df["prediction_start_date"] = pd.to_datetime(processed_df["prediction_start_date"])

        return processed_df

    @staticmethod
    def add_calender_features(dataframe, calender_data):
        assert "date" in calender_data.columns
        calender_data["prediction_start_date"] = pd.to_datetime(calender_data["date"])
        dataframe = pd.merge(dataframe, calender_data, on="prediction_start_date")
        assert (dataframe["prediction_start_date"] == dataframe["date"]).all()
        return dataframe



# This function is called from the main class.
if __name__ == "__main__":
    preprocessing_parameters = {"state_id":"WI",
                                "horizon": 28,
                                "lookback_multiple": 2,
                                "raw_data_path": "src/data/raw",
                                "bronze_data_path": "src/data/bronze",
                                "processed_file_name": "m5_processed"}
    preprocessor = Preprocess(preprocessing_parameters)
    preprocessor.main()
