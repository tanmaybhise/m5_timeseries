import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from main.utils import utils
from main.utils.logger import logger as logging
from config import Config

from datetime import datetime
import pandas as pd
import pickle
import os

class M5model():
    def __init__(self, parameters) -> None:
        self.silver_data_path = Config.silver_data_path
        self.features_data_name = parameters["features_data_name"]
        self.trained_models_path = Config.trained_models_path
        self.gold_data_path = Config.gold_data_path
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
            pickle.dump(trained_model, open(f"{self.trained_models_path}/{model}_model.pkl", "wb"))
            logging.info(f"Training {model} finished at {datetime.now()}")
        
        logging.info(f"Training process finished at {datetime.now()}")

    def infer(self):
        utils.make_dir_if_not_exist(self.gold_data_path)
        logging.info(f"Inference process started at {datetime.now()}")
        data = self.extract()
        features = pd.read_csv(f"{self.trained_models_path}/features.csv", index_col=None).values.flatten()
        for model in self.models:
            loaded_model = pickle.load(open(f"{self.trained_models_path}/{model}_model.pkl", "rb"))
            pred = loaded_model.predict(data.loc[:, features]).astype(int)
            pred_df = pd.DataFrame(pred, columns=[f"F{n}" for n in range(1,29)])
            pred_df["id"] = data["id"].values.flatten()
            final_columns = ["id"]+[f"F{n}" for n in range(1,29)]
            pred_df[final_columns].to_csv(f"{self.gold_data_path}/{model}_predictions_{self.prediction_data_suffix}.csv", index=None)
        
        logging.info(f"Inference process finished at {datetime.now()}")

    def create_submision(self):
        logging.info(f"Creating submission started at {datetime.now()}")
        for model in self.models:
            pred_evaluation = pd.read_csv(f"{self.gold_data_path}/{model}_predictions_validation.csv")
            pred_validation = pd.read_csv(f"{self.gold_data_path}/{model}_predictions_evaluation.csv")
            pd.concat([pred_validation, pred_evaluation]).to_csv(f"{self.gold_data_path}/{model}_submission.csv", index=None)
            os.remove(f"{self.gold_data_path}/{model}_predictions_validation.csv")
            os.remove(f"{self.gold_data_path}/{model}_predictions_evaluation.csv")
        logging.info(f"Creating submission finished at {datetime.now()}")

            
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
    model_parameters = {"features_data_name": "features",
                        "models": "linear,lgb",
                        "mode": "train"
                        }
    model = M5model(parameters=model_parameters)
    model.train()