import os
from math import sqrt

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def cross_validate_model(x, y, model):
    tscv = TimeSeriesSplit(n_splits=5, test_size=5000)

    mse = 0
    #acc, prec, rec, f1 = 0, 0, 0, 0

    for k, (train_index, test_index) in enumerate(tscv.split(x)):
        # Split dataset
        x_train = x.loc[train_index]
        x_test = x.loc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # Train model
        model_cv = model.fit(x_train, y_train)

        # Evaluate model
        y_pred = model_cv.predict(x_test)
        #acc += accuracy_score(y_test, y_pred)
        #prec += precision_score(y_test, y_pred)
        #rec += recall_score(y_test, y_pred)
        #f1 += f1_score(y_test, y_pred)
        mse += mean_squared_error(y_test, y_pred)

    #acc = acc / 5
    #prec = prec / 5
    #rec = rec / 5
    #f1 = f1 / 5
    mse /= 5
    return mse


# Load data
df = pd.read_csv('./data/etl/features.csv')

# Choose features
columns = ['Neutral', 'IsHome', 'IsMajorTournament', 'IsFriendly', 'IsEuros', 'Year',
           'Recent3', 'Recent5', 'Recent10', 'EloA', 'EloB']
df_x = df[columns]
df_y = df['ScoreA']

# Cross validate and fit model
model  = XGBRegressor()
mse = cross_validate_model(df_x, df_y, model)
rmse = sqrt(mse)
print(f'Mean Squared Error: {round(rmse, 5)}')
model = XGBRegressor().fit(df_x, df_y)

# Save model
os.makedirs('./data/models/', exist_ok=True)
model.save_model('./data/models/supercomputer.model')
