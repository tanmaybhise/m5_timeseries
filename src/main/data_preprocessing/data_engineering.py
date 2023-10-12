import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append("src")
import zipfile
import os
from datetime import datetime, timedelta
from collections import namedtuple
import numpy as np
import pandas as pd

from main.utils import utils
from main.utils.logger import logger as logging
from main.config import Config
from tqdm import tqdm

class Preprocess():
    def __init__(self, parameters):
        """
         Initialize the instance. This is called by the __init__ method of the : class : ` Horizon ` class
         
         Args:
         	 parameters: Dictionary containing the parameters
        """
        self.state_id = Config.state_id.split(",")
        self.horizon = Config.horizon
        self.lookback_multiple = Config.lookback_multiple
        self.raw_data_path = Config.raw_data_path
        self.bronze_data_path = Config.bronze_data_path
        self.raw_data_name = parameters["raw_data_name"]
        self.processed_file_name = parameters["processed_file_name"] 
        self.mode = parameters["mode"] 

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

        dataframe = pd.read_csv(self.raw_data_path+f"/{self.raw_data_name}.csv")

        logging.info(f"Extract process finished at {datetime.now()}")
        return dataframe

    def transform(self, dataframe):
        """
         Transforms data to be used in preprocessing. This is a method to be called by the transform_to_data method
         
         Returns: 
         	 a list of data
        """
        logging.info(f"Transform process started at {datetime.now()}")
        dataframe = dataframe[dataframe["state_id"].isin(self.state_id)]
        static_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        if self.mode == "train":
            drop_days = np.array([f"d_{n}" for n in range(1, Config.processing_start_day)])
            dataframe = dataframe.drop(drop_days, axis=1).reset_index(drop=True)
        elif self.mode == "infer":
            day_codes = dataframe.columns[dataframe.columns.str.contains("d_")]
            max_day = max([int(day_code.split("_")[-1]) for day_code in day_codes])
            drop_days = [f"d_{i}" for i in range(1, max_day - (self.horizon*self.lookback_multiple - 1))]
            dataframe = dataframe.drop(drop_days, axis=1).reset_index(drop=True)
            pred_columns = [f"d_{n}" for n in range(max_day+1, max_day+self.horizon+1)]
            dataframe.loc[:, pred_columns] = -9999
        else:
            raise NotImplementedError(f"Filtering for mode {self.mode} is not implemented")
        dataframe = dataframe.melt(id_vars=static_columns, 
                                   value_name="demand", var_name="d")
        preprocessed_data = self.add_leads_and_lags_features(dataframe, 
                                                column_name="demand", 
                                                group_by=["d"],
                                                lags=np.arange(1, 
                                                    Config.horizon*Config.lookback_multiple+1),
                                                leads=np.arange(1, Config.horizon))
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
    def add_leads_and_lags_features(dataframe, 
                                    column_name="demand", 
                                    group_by=["day"],
                                    lags=np.arange(1, 
                                            Config.horizon*Config.lookback_multiple+1),
                                    leads=np.arange(1, Config.horizon)):
        
        def _add_shift_features(dataframe, column_name, 
                                group_by, shifts):
            for shift in shifts:
                if shift !=0:
                    if shift > 0:
                        new_column_name=f"{column_name}_lag_{shift}"
                    elif shift < 0:
                        new_column_name=f"target_{abs(shift)+1}"
                    dataframe.loc[:, new_column_name] = \
                                    dataframe.groupby(group_by)[column_name]\
                                    .transform(lambda df: df.shift(shift))
            return dataframe
        
        dataframe = _add_shift_features(dataframe, column_name=column_name, 
                                             group_by=group_by, shifts=np.array(lags))
        dataframe = _add_shift_features(dataframe, column_name=column_name, 
                                             group_by=group_by, shifts=-1*(np.array(leads)-1))
        dataframe = dataframe.rename(columns = {column_name: "target_1"})
        dataframe = dataframe.dropna(axis=0).reset_index(drop=True)
        return dataframe

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
    def add_calender_features(dataframe, calender_data):
        assert "date" in calender_data.columns
        calender_data["prediction_start_date"] = pd.to_datetime(calender_data["date"])
        dataframe = pd.merge(dataframe, calender_data, on="d")
        assert (dataframe["prediction_start_date"] == dataframe["date"]).all()
        return dataframe



# This function is called from the main class.
if __name__ == "__main__":
    preprocessing_parameters = {"raw_data_name": "sales_train_evaluation",
                                "processed_file_name": f"m5_processed_infer",
                                "mode": "infer"}
    preprocessor = Preprocess(preprocessing_parameters)
    preprocessor.main()
