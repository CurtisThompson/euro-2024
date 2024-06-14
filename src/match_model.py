import pickle
from sklearn.base import BaseEstimator
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


class MatchModel(BaseEstimator):

    def __init__(self, **kwargs):
        #self.model_a = XGBRegressor()#XGBRegressor(**kwargs)
        #self.model_b = XGBRegressor()#XGBRegressor(**kwargs)
        #self.scaler = StandardScaler()
        self.model = MultiOutputRegressor(XGBRegressor(**kwargs))
    

    def fit(self, x, ya, yb):
        # Fit models
        #self.model_a = self.model_a.fit(x, ya)
        #self.model_b = self.model_b.fit(x, yb)
        y = np.array([ya, yb]).T
        #self.scaler = self.scaler.fit(x)
        #x_temp = self.scaler.transform(x)
        #x_temp[:,19] *= 5
        self.model = self.model.fit(x, y)

        return self
    

    def predict(self, x):
        # Get predictions
        #pred_a = np.round(self.model_a.predict(x)).astype(int)
        #pred_b = np.round(self.model_b.predict(x)).astype(int)
        #x_temp = self.scaler.transform(x)
        #x_temp[:,19] *= 5
        preds = np.round(self.model.predict(x)).astype(int)

        return preds#np.array([pred_a, pred_b]).T
    

    def simulate(self, x):
        # Get predictions
        #pred_a = self.model_a.predict(x)
        #pred_b = self.model_b.predict(x)
        preds = self.predict(x)
        pred_a = preds[:,0]
        pred_b = preds[:,1]

        # Add variance
        pred_a = pred_a + np.random.normal(loc=0, scale=0.1, size=len(pred_a))
        pred_b = pred_b + np.random.normal(loc=0, scale=0.1, size=len(pred_b))

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
