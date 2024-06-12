import pickle
from sklearn.base import BaseEstimator
import numpy as np
from xgboost import XGBRegressor


class MatchModel(BaseEstimator):

    def __init__(self, **kwargs):
        self.model_a = XGBRegressor(**kwargs)
        self.model_b = XGBRegressor(**kwargs)
    

    def fit(self, x, ya, yb):
        # Fit models
        self.model_a = self.model_a.fit(x, ya)
        self.model_b = self.model_b.fit(x, yb)

        return self
    

    def predict(self, x):
        # Get predictions
        pred_a = np.round(self.model_a.predict(x)).astype(int)
        pred_b = np.round(self.model_b.predict(x)).astype(int)

        return np.array([pred_a, pred_b]).T
    

    def save_model(self, path):
        pickle.dump(self, open(path, 'wb'))
    

    def load_model(self, path):
        return pickle.load(open(path, 'rb'))
