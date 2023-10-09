import sys
sys.path.append("src")
from main.data_preprocessing.data_engineering import Preprocess
from main.data_preprocessing.feature_engineering import FeatureEngineering
from main.config import Config
from main.utils.logger import logger as logging
from modeling.m5_model import M5model

if len(sys.argv) > 1:
    preprocessing_parameters, feature_engineering_parameters, model_parameters = Config.get_config(sys.argv[1])
    if sys.argv[1]=="train":
        logging.info("Executing script for training")
        preprocessor = Preprocess(parameters=preprocessing_parameters)
        preprocessor.main()

        feature_engineering = FeatureEngineering(parameters=feature_engineering_parameters)
        feature_engineering.main()
        model = M5model(parameters=model_parameters)
        model.train()
    elif sys.argv[1]=="infer":
        logging.info("Executing script for inference")
        for dataset in Config.datasets:

            preprocessing_parameters["raw_data_name"] = dataset[1]
            model_parameters["prediction_data_suffix"] = dataset[0]

            preprocessor = Preprocess(parameters=preprocessing_parameters)
            preprocessor.main()

            feature_engineering = FeatureEngineering(parameters=feature_engineering_parameters)
            feature_engineering.main()
            model = M5model(parameters=model_parameters)
            model.infer()
        model.create_submision()
else:
    preprocessing_parameters, feature_engineering_parameters, model_parameters = Config.get_config("train")
    logging.info("Executing script for data processing only")
    preprocessor = Preprocess(parameters=preprocessing_parameters)
    preprocessor.main()

    feature_engineering = FeatureEngineering(parameters=feature_engineering_parameters)
    feature_engineering.main()
