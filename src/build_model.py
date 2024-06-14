import os
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from match_model import MatchModel


def cross_validate_model(x, ya, yb, model):
    tscv = TimeSeriesSplit(n_splits=5, test_size=5000)

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

    mse /= 5
    mse_a /= 5
    mse_b /= 5
    mse_diff /= 5
    draws /= 5
    return mse, mse_a, mse_b, mse_diff, draws


def custom_dual_mse_loss(mse_a, mse_b, mse_diff, draws):
    return (mse_a + mse_b + draws) / 3


def find_best_hyperparameters(df_x, df_ya, df_yb, max_evals=100):
    def objective(space):
        model = MatchModel(**space)
        mse, mse_a, mse_b, mse_diff, draws = cross_validate_model(df_x, df_ya, df_yb, model)
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
        'seed' : 0
    }

    # Find best hyperparameters
    trials = Trials()
    best_hyperparameters = fmin(fn=objective,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=max_evals,
                                trials=trials)
    print('RMSE:', min(trials.losses()))
    print(best_hyperparameters)
    return best_hyperparameters


# Load data
df = pd.read_csv('./data/etl/features.csv')

# Choose features
columns = ['Neutral', 'IsHomeA', 'IsHomeB', 'IsMajorTournament', 'IsFriendly', 'IsEuros', 'Year',
           'Recent3A', 'Recent5A', 'Recent10A', 'Recent3B', 'Recent5B', 'Recent10B',
           'RecentGF10A', 'RecentGA10A', 'RecentGF10B', 'RecentGA10B', 'EloA', 'EloB', 'EloDiff']
df_x = df[columns]
df_ya = df['ScoreA']
df_yb = df['ScoreB']

# Cross validate and fit model
tune_model = False
if tune_model:
    params = find_best_hyperparameters(df_x, df_ya, df_yb)
    model = MatchModel(**params).fit(df_x, df_ya, df_yb)
else:
    params = {'colsample_bytree': 0.9968363907025145,
              'gamma': 3.7964109350382604,
              'max_depth': 4,
              'min_child_weight': 5.046692046533741,
              'n_estimators': 275,
              'reg_alpha': 14.716401696781382,
              'reg_lambda': 0.4974731068232508}
    params_2 = {'colsample_bytree': 0.8303200756641117, 
                'gamma': 1.2532646140877886,
                'max_depth': 3,
                'min_child_weight': 6.846057210847509,
                'n_estimators': 467,
                'reg_alpha': 24.62822865873158,
                'reg_lambda': 0.016366596430151065}
    model = MatchModel(**params_2)
    mse, mse_a, mse_b, mse_diff, draws = cross_validate_model(df_x, df_ya, df_yb, model)
    rmse = sqrt(mse)
    loss = custom_dual_mse_loss(mse_a, mse_b, mse_diff, draws)
    print(f'Mean Squared Error: {round(rmse, 5)}')
    print(f'Loss: {round(loss, 5)}')

# Save model
os.makedirs('./data/models/', exist_ok=True)
model.save_model('./data/models/supercomputer.model')
