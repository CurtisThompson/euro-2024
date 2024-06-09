import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def cross_validate_model(x, y, model):
    tscv = TimeSeriesSplit(n_splits=5, test_size=5000)

    acc, prec, rec, f1 = 0, 0, 0, 0

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
        acc += accuracy_score(y_test, y_pred)
        prec += precision_score(y_test, y_pred)
        rec += recall_score(y_test, y_pred)
        f1 += f1_score(y_test, y_pred)

    acc = acc / 5
    prec = prec / 5
    rec = rec / 5
    f1 = f1 / 5
    return acc, prec, rec, f1


# Load data
df = pd.read_csv('./data/etl/features.csv')

# Choose features
columns = ['Neutral', 'IsHome', 'IsMajorTournament', 'IsFriendly', 'IsEuros', 'Year',
           'Recent3', 'Recent5', 'Recent10', 'EloA', 'EloB']
df_x = df[columns]
df_y = df.Points == 3

# Cross validate and fit model
model  = XGBClassifier()
acc, prec, rec, f1 = cross_validate_model(df_x, df_y, model)
print(f'Accuracy: {round(acc, 5)}, Precision: {round(prec, 5)}, Recall: {round(rec, 5)}, F1 Score: {round(f1, 5)}')
model = XGBClassifier().fit(df_x, df_y)

# Test prediction
#df_pred = pd.DataFrame([[1, 0, 1, 0, 1, 2024, 3, 3, 3, 1600, 1500]], columns=columns)
#df_pred['Pred'] = model.predict_proba(df_pred)[:,1]

#print(df_pred)