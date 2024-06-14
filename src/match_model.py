import pickle
import numpy as np
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor


class MatchModel(BaseEstimator):
    """A machine learning model to predict the output of football matches.
    The model has two outputs for the home and away scores."""

    def __init__(self, **kwargs):
        """Constructor method"""
        self.model = MultiOutputRegressor(XGBRegressor(**kwargs))
    

    def fit(self, x, ya, yb):
        """Fit the model to data

        :param x: Input data
        :param ya: Home scores
        :param yb: Away scores

        :type x: Pandas DataFrame
        :type ya: Array-like of int
        :type yb: Array-like of int

        :return: Fitted model
        :rtype: MatchModel
        """
        # Fit models
        y = np.array([ya, yb]).T
        self.model = self.model.fit(x, y)

        return self
    

    def predict(self, x):
        """Predict match score (without added randomness).

        :param x: Features to predict on

        :type x: Pandas DataFrame

        :return: Score predictions
        :rtype: NumPy array
        """
        preds = np.round(self.model.predict(x)).astype(int)
        return preds
    

    def simulate(self, x):
        """Predict match score with added randomness

        :param x: Features to predict on

        :type x: Pandas DataFrame

        :return: Score predictions
        :rtype: NumPy array
        """
        # Get predictions
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
        """Save model to file.

        :param path: File path to model
        
        :type path: str
        """
        pickle.dump(self, open(path, 'wb'))
    

    def load_model(self, path):
        """Load model from file.

        :param path: File path to model
        
        :type path: str

        :return: Loaded model
        :rtype: MatchModel
        """
        return pickle.load(open(path, 'rb'))
