import os
from math import sqrt

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from match_model import MatchModel


def cross_validate_model(x, ya, yb, model):
    tscv = TimeSeriesSplit(n_splits=5, test_size=5000)

    mse = 0
    #acc, prec, rec, f1 = 0, 0, 0, 0

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
        ya_pred, yb_pred = model_cv.predict(x_test)
        mse += (mean_squared_error(ya_test, ya_pred) + mean_squared_error(yb_test, yb_pred)) / 2

    mse /= 5
    return mse


# Load data
df = pd.read_csv('./data/etl/features.csv')

# Choose features
columns = ['Neutral', 'IsHomeA', 'IsHomeB', 'IsMajorTournament', 'IsFriendly', 'IsEuros', 'Year',
           'Recent3A', 'Recent5A', 'Recent10A', 'Recent3B', 'Recent5B', 'Recent10B', 'EloA', 'EloB']
df_x = df[columns]
df_ya = df['ScoreA']
df_yb = df['ScoreB']

# Cross validate and fit model
model = MatchModel()
mse = cross_validate_model(df_x, df_ya, df_yb, model)
rmse = sqrt(mse)
print(f'Mean Squared Error: {round(rmse, 5)}')
model = MatchModel().fit(df_x, df_ya, df_yb)

# Save model
os.makedirs('./data/models/', exist_ok=True)
model.save_model('./data/models/supercomputer.model')
