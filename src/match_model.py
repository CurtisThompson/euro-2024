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
    

    def simulate(self, x):
        # Get predictions
        pred_a = self.model_a.predict(x)
        pred_b = self.model_b.predict(x)

        # Add variance
        pred_a = pred_a + np.random.normal(loc=0, scale=0.3, size=len(pred_a))
        pred_b = pred_a + np.random.normal(loc=0, scale=0.3, size=len(pred_b))

        # Tidy preds
        pred_a[pred_a < 0] = 0
        pred_a = np.round(pred_a).astype(int)
        pred_b[pred_a < 0] = 0
        pred_b = np.round(pred_b).astype(int)

        return np.array([pred_a, pred_b]).T
    

    def save_model(self, path):
        pickle.dump(self, open(path, 'wb'))
    

    def load_model(self, path):
        return pickle.load(open(path, 'rb'))
