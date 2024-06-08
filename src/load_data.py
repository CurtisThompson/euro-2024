import os

import pandas as pd
import numpy as np


# Load results
df = pd.read_csv('./data/raw/results.csv')

# Change column names
df.columns = ['Date', 'TeamA', 'TeamB', 'ScoreA', 'ScoreB', 'Tournament', 'City', 'Country', 'Neutral']

# Drop NaN matches (missing data or not played yet)
df = df.dropna(axis=0).reset_index(drop=True)

# Duplicate rows and switch
df_switched = df.copy()
df_switched[['TeamA', 'TeamB', 'ScoreA', 'ScoreB']] = df_switched[['TeamB', 'TeamA', 'ScoreB', 'ScoreA']]
df['IsHome'] = 1
df_switched['IsHome'] = 0
df = pd.concat([df, df_switched]).sort_values('Date').reset_index(drop=True)

# Set correct types for loaded columns
df['ScoreA'] = df['ScoreA'].astype('int')
df['ScoreB'] = df['ScoreB'].astype('int')
df['Neutral'] = df['Neutral'].astype('int')

# Generate tournament-related features
major_tournaments = ['FIFA World Cup', 'Copa América', 'African Cup of Nations', 'AFC Asian Cup',
                     'CONCACAF Championship', 'UEFA Euro']
df['IsMajorTournament'] = df.Tournament.isin(major_tournaments).astype('int')
df['IsFriendly'] = df.Tournament.eq('Friendly').astype('int')
df['IsEuros'] = df.Tournament.eq('UEFA Euro').astype('int')

# Generate date features
df['Year'] = pd.DatetimeIndex(df.Date).year

# Get match order
df = df.reset_index().rename({'index' : 'MatchOrder'}, axis=1)

# Get form-related features
df['Points'] = (3 * (df.ScoreA > df.ScoreB)) + (df.ScoreA == df.ScoreB)
df['Recent3'] = df.groupby('TeamA').rolling(3).Points.sum().droplevel(0).fillna(0).astype(int) - df.Points
df['Recent5'] = df.groupby('TeamA').rolling(5).Points.sum().droplevel(0).fillna(0).astype(int) - df.Points
df['Recent10'] = df.groupby('TeamA').rolling(10).Points.sum().droplevel(0).fillna(0).astype(int) - df.Points

# Save features
os.makedirs('./data/etl/', exist_ok=True)
df.to_csv('./data/etl/features.csv', index=False)