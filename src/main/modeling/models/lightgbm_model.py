from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator
import lightgbm

class LGBModel(BaseEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.model = MultiOutputRegressor(
                        lightgbm.LGBMRegressor(
                            objective="tweedie"))

    def fit(self, x,y):
        self.model.fit(x, y)
        return self
    
    def predict(self, x):
        yhat = self.model.predict(x)
        return yhat
