import os
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from match_model import MatchModel


def cross_validate_model(x, ya, yb, model, n_splits=5):
    """Time series split validation of given model.
    
    :param x: Model input data
    :param ya: Model true output (home scores)
    :param yb: Model true output (away scores)
    :param model: Machine learning model to evaluate
    :param n_splits: Number of time series splits, defaults to 5

    :type x: Pandas DataFrame
    :type ya: Array-like
    :type yb: Array-like
    :type model: MatchModel
    :type n_splits: int

    :return: Mean squared errors combined, for ya, for yb, mean squared error
        of score difference, and percentage of games predicted as draws
    :rtype: float
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=5000)

    mse, mse_a, mse_b, mse_diff, draws = 0, 0, 0, 0, 0

    for k, (train_index, test_index) in enumerate(tscv.split(x)):
        # Split dataset
        x_train = x.loc[train_index]
        x_test = x.loc[test_index]
        ya_train = ya[train_index]
        ya_test = ya[test_index]
        yb_train = yb[train_index]
        yb_test = yb[test_index]

        # Train model
        model_cv = model.fit(x_train, ya_train, yb_train)

        # Evaluate model
        ya_pred, yb_pred = np.array(model_cv.predict(x_test)).T
        mse_a += mean_squared_error(ya_test, ya_pred)
        mse_b += mean_squared_error(yb_test, yb_pred)
        mse_diff += mean_squared_error(ya_test - yb_test, ya_pred - yb_pred)
        mse += (mean_squared_error(ya_test, ya_pred) + mean_squared_error(yb_test, yb_pred)) / 2
        draws += (sum(ya_pred == yb_pred) / len(ya_pred))

    mse /= n_splits
    mse_a /= n_splits
    mse_b /= n_splits
    mse_diff /= n_splits
    draws /= n_splits
    return mse, mse_a, mse_b, mse_diff, draws


def custom_dual_mse_loss(mse_a, mse_b, mse_diff, draws):
    """Custom loss function, average of mean squared error and draws.
    
    :param mse_a: Mean squared error of home scores
    :param mse_b: Mean squared error of away scores
    :param mse_diff: Mean squared error of score difference
    :param draws: Percentage of draws

    :type mse_a: float
    :type mse_b: float
    :type mse_diff: float
    :type draws: float

    :return: loss
    :rtype: float
    """
    return (mse_a + mse_b + draws) / 3


def find_best_hyperparameters(df_x, df_ya, df_yb, max_evals=100, verbose=True):
    """Hyperparameter search of model.

    :param df_x: Model input data
    :param df_ya: Model true output (home scores)
    :param df_yb: Model true output (away scores)
    :param max_evals: Maximum number of models to evaluate, defaults to 100
    :param verbose: Whether to output progress, defaults to True

    :type df_x: Pandas DataFrame
    :type df_ya: Array-like
    :type df_yb: Array-like
    :type max_evals: int
    :type verbose: bool

    :return: Best hyperparameters found
    :rtype: dict
    """
    def objective(space):
        model = MatchModel(**space)
        mse, mse_a, mse_b, mse_diff, draws = cross_validate_model(df_x,
                                                                  df_ya,
                                                                  df_yb, 
                                                                  model)
        loss = custom_dual_mse_loss(mse_a, mse_b, mse_diff, draws)
        return {'loss': loss, 'status': STATUS_OK}

    # Hyperparameter space
    space = {
        'max_depth' : hp.uniformint('max_depth', 3, 19),
        'gamma' : hp.uniform('gamma', 1, 9),
        'reg_alpha' : hp.uniform('reg_alpha', 1, 100),
        'reg_lambda' : hp.uniform('reg_lambda', 0, 1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
        'min_child_weight' : hp.uniform('min_child_weight', 0, 10,),
        'n_estimators' : hp.uniformint('n_estimators', 10, 1000),
        'random_state' : 0
    }

    # Find best hyperparameters
    trials = Trials()
    best_hyperparameters = fmin(fn=objective,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=max_evals,
                                trials=trials)
    if verbose:
        print('RMSE:', min(trials.losses()))
        print(best_hyperparameters)
    return best_hyperparameters


def build_model(verbose=True):
    """Load training data, fit model, and save to file.
    
    :param verbose: Whether to output progress, defaults to True
    
    :type verbose: bool"""
    
    # Load data
    df = pd.read_csv('./data/etl/features.csv')

    # Choose features
    columns = ['Neutral', 'IsHomeA', 'IsHomeB', 'IsMajorTournament',
               'IsFriendly', 'IsEuros', 'Year', 'Recent3A', 'Recent5A',
               'Recent10A', 'Recent3B', 'Recent5B', 'Recent10B', 'RecentGF10A',
               'RecentGA10A', 'RecentGF10B', 'RecentGA10B',
               'EloA', 'EloB', 'EloDiff']
    df_x = df[columns]
    df_ya = df['ScoreA']
    df_yb = df['ScoreB']

    # Cross validate and fit model
    tune_model = False
    if tune_model:
        params = find_best_hyperparameters(df_x, df_ya, df_yb, verbose=verbose)
        model = MatchModel(**params).fit(df_x, df_ya, df_yb)
    else:
        params = {'colsample_bytree': 0.8303200756641117, 
                  'gamma': 1.2532646140877886,
                  'max_depth': 3,
                  'min_child_weight': 6.846057210847509,
                  'n_estimators': 467,
                  'reg_alpha': 24.62822865873158,
                  'reg_lambda': 0.016366596430151065}
        model = MatchModel(**params)
        mse, mse_a, mse_b, mse_diff, draws = cross_validate_model(df_x, df_ya, df_yb, model)
        rmse = sqrt(mse)
        loss = custom_dual_mse_loss(mse_a, mse_b, mse_diff, draws)
        if verbose:
            print(f'Root Mean Squared Error: {round(rmse, 5)}')
            print(f'Loss: {round(loss, 5)}')

    # Save model
    os.makedirs('./data/models/', exist_ok=True)
    model.save_model('./data/models/supercomputer.model')
