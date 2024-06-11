import os
import json

import pandas as pd
import numpy as np


def calculate_elo(df, K=32):
    df_once = df.sort_values('Date', ascending=True).drop_duplicates(subset=['Date', 'MatchNumber'], keep='first')
    elos = dict.fromkeys(df['TeamA'].unique(), 1500)

    df['EloA'] = 0
    df['EloB'] = 0

    for index, game in df_once.iterrows():
        if index % 2000 == 0:
            print(index)
        
        # Set pre-Elo value in main dataframe
        df.loc[index, 'EloA'] = elos[game.TeamA]
        df.loc[index, 'EloB'] = elos[game.TeamB]

        # Get team IDs
        teamA = game.TeamA
        teamB = game.TeamB

        # Get winner
        winA = 1 if game.PointsA == 3 else 0.5 if game.PointsA == 1 else 0
        winB = 1 - winA

        # Get ELO stats
        eloA = elos[teamA]
        eloB = elos[teamB]

        # Calc ELO changes
        rA = 10 ** (eloA / 400)
        rB = 10 ** (eloB / 400)
        eA = rA / (rA + rB)
        eB = rB / (rA + rB)
        sA = winA
        sB = winB
        newEloA = eloA + (K * (sA - eA))
        newEloB = eloB + (K * (sB - eB))

        # Update ELO frame
        elos[teamA] = newEloA
        elos[teamB] = newEloB
    
    # Join in reverse of calculated Elos
    df_sub = df[['MatchNumber', 'TeamB', 'EloA', 'EloB']]
    df = df.merge(df_sub, left_on=['MatchNumber', 'TeamA'], right_on=['MatchNumber', 'TeamB'], suffixes=['', '_remove'])
    df['EloA'] = df[['EloA', 'EloB_remove']].values.max(1)
    df['EloB'] = df[['EloB', 'EloA_remove']].values.max(1)
    df = df.drop(['TeamB_remove', 'EloA_remove', 'EloB_remove'], axis=1)
    return df, elos


# Load results
df = pd.read_csv('./data/raw/results.csv')

# Change column names
df.columns = ['Date', 'TeamA', 'TeamB', 'ScoreA', 'ScoreB', 'Tournament', 'City', 'Country', 'Neutral']

# Drop NaN matches (missing data or not played yet)
df = df.dropna(axis=0).reset_index(drop=True)

# Duplicate rows and switch
df = df.reset_index(drop=False).rename({'index' : 'MatchNumber'}, axis=1)
df_switched = df.copy()
df_switched[['TeamA', 'TeamB', 'ScoreA', 'ScoreB']] = df_switched[['TeamB', 'TeamA', 'ScoreB', 'ScoreA']]
df['IsHomeA'] = 1
df['IsHomeB'] = 0
df_switched['IsHomeA'] = 0
df_switched['IsHomeB'] = 1
df = pd.concat([df, df_switched]).sort_values('Date').reset_index(drop=True)
df = df.reset_index().rename({'index' : 'MatchOrder'}, axis=1)

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

# Get form-related features
df['PointsA'] = (3 * (df.ScoreA > df.ScoreB)) + (df.ScoreA == df.ScoreB)
df['PointsB'] = (3 * (df.ScoreB > df.ScoreA)) + (df.ScoreB == df.ScoreA)
df['Recent3A'] = df.groupby('TeamA').rolling(4).PointsA.sum().droplevel(0).fillna(0).astype(int) - df.PointsA
df['Recent5A'] = df.groupby('TeamA').rolling(6).PointsA.sum().droplevel(0).fillna(0).astype(int) - df.PointsA
df['Recent10A'] = df.groupby('TeamA').rolling(11).PointsA.sum().droplevel(0).fillna(0).astype(int) - df.PointsA
df['Recent3B'] = df.groupby('TeamB').rolling(4).PointsB.sum().droplevel(0).fillna(0).astype(int) - df.PointsB
df['Recent5B'] = df.groupby('TeamB').rolling(6).PointsB.sum().droplevel(0).fillna(0).astype(int) - df.PointsB
df['Recent10B'] = df.groupby('TeamB').rolling(11).PointsB.sum().droplevel(0).fillna(0).astype(int) - df.PointsB

df, elos = calculate_elo(df)

# Save features
os.makedirs('./data/etl/', exist_ok=True)
df.to_csv('./data/etl/features.csv', index=False)
with open('./data/etl/elo.json', 'w', encoding='utf-8') as f:
    json.dump(elos, f)