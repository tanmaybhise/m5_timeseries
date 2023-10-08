import sys
sys.path.append("src")
from main.data_preprocessing.data_engineering import Preprocess
from main.data_preprocessing.feature_engineering import FeatureEngineering
from main.parameters import Parameters

preprocessor = Preprocess(parameters=Parameters.preprocessing_parameters)
preprocessor.main()

feature_engineering = FeatureEngineering(parameters=Parameters.feature_engineering_parameters)
feature_engineering.main()

