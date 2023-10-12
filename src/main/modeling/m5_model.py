import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from main.utils import utils
from main.utils.logger import logger as logging
from main.config import Config

from datetime import datetime
import pandas as pd
import pickle
import os
from sklearn import metrics
from tqdm import tqdm

from main.data_preprocessing.data_engineering import Preprocess
from main.data_preprocessing.feature_engineering import FeatureEngineering

class M5model():
    def __init__(self, parameters):
        self.silver_data_path = Config.silver_data_path
        self.features_data_name = parameters["features_data_name"]
        self.trained_models_path = Config.trained_models_path
        self.gold_data_path = Config.gold_data_path
        try:
            self.raw_data_name = parameters["raw_data_name"]
        except Exception as e:
            logging.error(e)
        self.models = parameters["models"].split(",")
        try:
            self.prediction_data_suffix = parameters["prediction_data_suffix"]
        except KeyError:
            self.prediction_data_suffix = ""
    
    def extract(self):
        logging.info(f"Extract process started at {datetime.now()}")
        dataframe = pd.read_parquet(f"{self.silver_data_path}/{self.features_data_name}.parquet")
        logging.info(f"Extract process finished at {datetime.now()}")
        return dataframe
    
    def train(self):
        logging.info(f"Training process started at {datetime.now()}")
        data = self.extract()
        train_df, val_df, test_df = self.train_val_test_split(data, val_batches=2, 
                                        test_batches=2, 
                                        date_column="prediction_start_date")
        utils.make_dir_if_not_exist(self.trained_models_path)
        targets = train_df.columns[train_df.columns.str.contains("target")]
        pd.Series(targets, name="targets").to_csv(f"{self.trained_models_path}/targets.csv", index=None)
        features = train_df.columns[~((train_df.columns.str.contains("target")) \
                                      | (train_df.columns.isin(["prediction_start_date", "id"])))]
        pd.Series(features, name="features").to_csv(f"{self.trained_models_path}/features.csv", index=None)

        for model in self.models:
            logging.info(f"Training {model} started at {datetime.now()}")
            trained_model = Config.models_dict[model].fit(train_df[features], train_df[targets])
            yhat = trained_model.predict(test_df[features])
            metrics_dict = {
                "model": model,
                "mae": metrics.mean_absolute_error(test_df[targets], yhat),
                "mape": metrics.mean_absolute_percentage_error(test_df[targets], yhat),
                "mse": metrics.mean_squared_error(test_df[targets], yhat),
                "rmse": metrics.mean_squared_error(test_df[targets], yhat, squared=False)
            }
            self.__create_metrics_df(metrics_dict)
            pickle.dump(trained_model, open(f"{self.trained_models_path}/{model}_model.pkl", "wb"))
            logging.info(f"Training {model} finished at {datetime.now()}")
        
        logging.info(f"Training process finished at {datetime.now()}")

    def infer(self):
        utils.make_dir_if_not_exist(self.gold_data_path)
        logging.info(f"Inference process started at {datetime.now()}")
        features = pd.read_csv(f"{self.trained_models_path}/features.csv", index_col=None).values.flatten()
        logging.disabled = True
        for model in self.models:
            data = self.extract()
            results_df = data[["id"]].copy()
            for n in tqdm(range(Config.inference_horizon), desc=f"Performing inference process for {model}"):
                loaded_model = pickle.load(open(f"{self.trained_models_path}/{model}_model.pkl", "rb"))
                pred = loaded_model.predict(data.loc[:, features].apply(pd.to_numeric))
                pred_df = pd.DataFrame(pred, columns=[f"F{m+1}" for m in range(n, n+Config.horizon)])
                pred_df["id"] = data["id"].values.flatten()
                final_columns = ["id"]+[f"F{m+1}" for m in range(n,n+Config.horizon)]
                results_df = pd.merge(results_df, pred_df[final_columns], on="id", how="left")
                data = self.get_inference_df(results_df)
            results_df.to_csv(f"{self.gold_data_path}/{model}_predictions_{self.prediction_data_suffix}.csv", index=None)
        logging.disabled = False
        
        logging.info(f"Inference process finished at {datetime.now()}")

    def get_inference_df(self, results_df):
        preprocessing_parameters, feature_engineering_parameters, _ = Config.get_config("infer")
        raw = pd.read_csv(f"{Config.raw_data_path}/{self.raw_data_name}.csv")
        day_codes = raw.columns[raw.columns.str.contains("d_")]
        max_day = max([int(day_code.split("_")[-1]) for day_code in day_codes])
        pred_column_names = {f"F{m+1}":f"d_{n}" for m,n \
                            in enumerate(range(max_day+1, max_day+Config.horizon+1))}
        tmp_df = results_df.rename(columns=pred_column_names)

        preprocessor = Preprocess(preprocessing_parameters)
        fe = FeatureEngineering(feature_engineering_parameters)
        inference_df = pd.merge(raw, tmp_df, how="inner", on="id")
        inference_df = preprocessor.transform(inference_df)
        inference_df = fe.transform(inference_df)
        return inference_df

    def create_submision(self):
        logging.info(f"Creating submission started at {datetime.now()}")
        for model in self.models:
            pred_evaluation = pd.read_csv(f"{self.gold_data_path}/{model}_predictions_validation.csv")
            pred_validation = pd.read_csv(f"{self.gold_data_path}/{model}_predictions_evaluation.csv")
            pd.concat([pred_validation, pred_evaluation]).to_csv(f"{self.gold_data_path}/{model}_submission.csv", index=None)
            os.remove(f"{self.gold_data_path}/{model}_predictions_validation.csv")
            os.remove(f"{self.gold_data_path}/{model}_predictions_evaluation.csv")
        logging.info(f"Creating submission finished at {datetime.now()}")

    def __create_metrics_df(self, metrics_dict):
        if os.path.isfile(f"{self.gold_data_path}/metrics.csv"):
            metric_df = pd.read_csv(f"{self.gold_data_path}/metrics.csv")
            running_index = metric_df.index[-1]+1
        else:
            metric_df = pd.DataFrame()
            running_index=0
        tmp_metric_df = pd.DataFrame()
        tmp_metric_df.loc[running_index, "datetime"] = datetime.now()
        tmp_metric_df.loc[running_index, "horizon"] = Config.horizon
        tmp_metric_df.loc[running_index, "lookback_multiple"] = Config.lookback_multiple
        tmp_metric_df.loc[running_index, "state_id"] = Config.state_id
        tmp_metric_df.loc[running_index, "processing_start_day"] = Config.processing_start_day
        tmp_metric_df.loc[running_index, "lookback_multiple"] = Config.lookback_multiple
        for metric, value in metrics_dict.items():
            tmp_metric_df.loc[running_index, metric] = value

        metric_df = pd.concat([metric_df, tmp_metric_df], axis=0).reset_index(drop=True)
        metric_df.to_csv(f"{self.gold_data_path}/metrics.csv", index=None)
        
    @staticmethod
    def train_val_test_split(datafarme, val_batches=2, 
                             test_batches=2, 
                             date_column="prediction_start_date"):
        datafarme = datafarme.\
            sort_values(by=date_column).reset_index(drop=True)
        
        test_dates = datafarme[date_column]\
                        .drop_duplicates()\
                        .values[-test_batches:]
        val_dates = datafarme[date_column]\
                        .drop_duplicates()\
                        .values[-(val_batches+test_batches):-test_batches]

        train_df = datafarme[~(datafarme[date_column].isin(test_dates) \
                            | datafarme[date_column].isin(val_dates))]
        val_df = datafarme[datafarme[date_column].isin(val_dates)]
        test_df = datafarme[datafarme[date_column].isin(test_dates)]
        return train_df, val_df, test_df
    
if __name__=="__main__":
    model_parameters = {
                        "raw_data_name": "sales_train_evaluation",
                        "features_data_name": "features",
                        "models": "linear,lgb",
                        "mode": "train"
                        }
    model = M5model(parameters=model_parameters)
    model.train()