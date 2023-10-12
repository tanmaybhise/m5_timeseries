import os
from main.config import Config
import pandas as pd
from tqdm import tqdm

def make_dir_if_not_exist(path):
    os.makedirs(path, exist_ok=True)

def create_stats_encoding(data, features, mode, 
                          feature_engineering_artifacts_path="src/data/feature_engineering_artifacts"):
    artifact = "stats_encodings"
    if mode=="train":
        static_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        raw = pd.read_csv(f"{Config.raw_data_path}/sales_train_evaluation.csv")
        calendar = pd.read_csv(f"{Config.raw_data_path}/calendar.csv")
        keep_day_codes = [f"d_{n}" for n in range(1, Config.processing_start_day)]
        selected_columns = static_columns+keep_day_codes
        raw = raw[selected_columns]
        raw = raw.melt(id_vars=static_columns, value_name="demand", var_name="d")
        raw = raw.merge(calendar, on="d", how="left")
        raw["generated_id"] = raw["item_id"]+"_"+raw["store_id"]
        for feature in tqdm(features, desc="Generating stats categorical encoding"):
            group_by = list(set(["generated_id", feature]))
            encodings = raw.groupby(group_by)[["demand"]]\
                .mean()\
                .rename(columns={"demand": f"{feature}_mean_demand"})
            encodings = pd.merge(encodings, raw.groupby(group_by)[["demand"]]\
                                .max()\
                                .rename(columns={"demand": f"{feature}_max_demand"}), 
                                        left_index=True, right_index=True)
            encodings = pd.merge(encodings, raw.groupby(group_by)[["demand"]]\
                                .std()\
                                .rename(columns={"demand": f"{feature}_std_demand"}), 
                                        left_index=True, right_index=True)
            encodings = encodings.reset_index()
            make_dir_if_not_exist(f"{feature_engineering_artifacts_path}/{artifact}")
            encodings.to_parquet(f"{feature_engineering_artifacts_path}/{artifact}/{feature}_stats_encoding.parquet")
            data = data.merge(encodings, how="left", on=group_by)
    elif mode=="infer":
        for feature in tqdm(features, desc="Generating stats categorical encoding"):
            group_by = list(set(["generated_id", feature]))
            encodings = pd.read_parquet(f"{feature_engineering_artifacts_path}/{artifact}/{feature}_stats_encoding.parquet")
            data = data.merge(encodings, how="left", on=group_by)
    return data.drop(features, axis=1)