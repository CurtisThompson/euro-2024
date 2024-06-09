import pandas as pd
from xgboost import XGBClassifier



# Load data
df = pd.read_csv('./data/etl/features.csv')

# Choose features
columns = ['Neutral', 'IsHome', 'IsMajorTournament', 'IsFriendly', 'IsEuros', 'Year',
           'Recent3', 'Recent5', 'Recent10', 'EloA', 'EloB']
df_x = df[columns]
df_y = df.Points == 3

# Fit model
model = XGBClassifier().fit(df_x, df_y)

# Test prediction
df_pred = pd.DataFrame([[1, 0, 1, 0, 1, 2024, 3, 3, 3, 1600, 1500]], columns=columns)
df_pred['Pred'] = model.predict_proba(df_pred)[:,1]

print(df_pred)