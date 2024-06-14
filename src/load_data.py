import os
import json

import pandas as pd


def calculate_elo(df, K=32, verbose=True):
    """Add Elo columns for both teams in a set of fixtures.

    :param df: Set of fixtures containing TeamA, TeamB, PointsA, MatchNumber
        and Date columns
    :param K: Elo K-factor, defaults to 32
    :param verbose: whether to output progress

    :type df: Pandas DataFrame
    :type K: int
    :type verbose: bool

    :return: Modified DataFrame with EloA and EloB columns.
    :rtype: Pandas DataFrame.
    """
    df_once = (
        df.sort_values('Date', ascending=True)
        .drop_duplicates(subset=['Date', 'MatchNumber'], keep='first')
    )
    elos = dict.fromkeys(df['TeamA'].unique(), 1500)
    df['EloA'] = 0
    df['EloB'] = 0

    # Iterate through games in order, calculating Elo one-at-a-time
    for index, game in df_once.iterrows():
        if (verbose) and (index % 2000 == 0):
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
    df = df.merge(
        df_sub,
        left_on=['MatchNumber', 'TeamA'],
        right_on=['MatchNumber', 'TeamB'],
        suffixes=['', '_remove']
    )
    df['EloA'] = df[['EloA', 'EloB_remove']].values.max(1)
    df['EloB'] = df[['EloB', 'EloA_remove']].values.max(1)
    df = df.drop(['TeamB_remove', 'EloA_remove', 'EloB_remove'], axis=1)
    return df, elos


def rolling_sum_feature(df, group, window, feature):
    """Calculate an integer feature based on a rolling sum window.
    
    :param df: DataFrame to apply rolling window to
    :param group: Column name of group for rolling window
    :param window: Size of rolling window
    :param feature: Column name of feature to sum

    :type df: Pandas DataFrame
    :type group: str
    :type window: int
    :type feature: str

    :return: Calculated feature
    :rtype: Pandas Series
    """
    return (
        df.groupby(group)
        .rolling(window)[feature]
        .sum()
        .droplevel(0)
        .fillna(0)
        .astype(int)
    )


def process_results(verbose=True):
    """Load raw results csv and calculate related features.
    
    :param verbose: Whether to output progress
    
    :type verbose: bool
    """
    # Load results
    df = pd.read_csv('./data/raw/results.csv')

    # Change column names
    df.columns = ['Date', 'TeamA', 'TeamB', 'ScoreA',
                  'ScoreB', 'Tournament', 'City', 'Country', 'Neutral']

    # Drop NaN matches (missing data or not played yet)
    df = df.dropna(axis=0).reset_index(drop=True)

    # Duplicate rows and switch
    df = df.reset_index(drop=False).rename({'index' : 'MatchNumber'}, axis=1)
    df_switched = df.copy()
    df_switched[['TeamA', 'TeamB', 'ScoreA', 'ScoreB']] = (
        df_switched[['TeamB', 'TeamA', 'ScoreB', 'ScoreA']]
    )
    df['IsHomeA'] = 1
    df['IsHomeB'] = 0
    df_switched['IsHomeA'] = 0
    df_switched['IsHomeB'] = 1
    df = (pd.concat([df, df_switched])
            .sort_values('Date').reset_index(drop=True)
            .reset_index().rename({'index' : 'MatchOrder'}, axis=1)
    )

    # Set correct types for loaded columns
    df['ScoreA'] = df['ScoreA'].astype('int')
    df['ScoreB'] = df['ScoreB'].astype('int')
    df['Neutral'] = df['Neutral'].astype('int')

    # Generate tournament-related features
    maj_tourns = ['FIFA World Cup', 'Copa América',
                  'African Cup of Nations', 'AFC Asian Cup',
                  'CONCACAF Championship', 'UEFA Euro']
    df['IsMajorTournament'] = df.Tournament.isin(maj_tourns).astype('int')
    df['IsFriendly'] = df.Tournament.eq('Friendly').astype('int')
    df['IsEuros'] = df.Tournament.eq('UEFA Euro').astype('int')

    # Generate date features
    df['Year'] = pd.DatetimeIndex(df.Date).year

    # Get form-related features
    df['PointsA'] = (3 * (df.ScoreA > df.ScoreB)) + (df.ScoreA == df.ScoreB)
    df['PointsB'] = (3 * (df.ScoreB > df.ScoreA)) + (df.ScoreB == df.ScoreA)
    df['Recent3A'] = rolling_sum_feature(df, 'TeamA', 4, 'PointsA') - df.PointsA
    df['Recent5A'] = rolling_sum_feature(df, 'TeamA', 6, 'PointsA') - df.PointsA
    df['Recent10A'] = rolling_sum_feature(df, 'TeamA', 11, 'PointsA') - df.PointsA
    df['Recent3B'] = rolling_sum_feature(df, 'TeamB', 4, 'PointsB') - df.PointsB
    df['Recent5B'] = rolling_sum_feature(df, 'TeamB', 6, 'PointsB') - df.PointsB
    df['Recent10B'] = rolling_sum_feature(df, 'TeamB', 11, 'PointsB') - df.PointsB
    df['RecentGF10A'] = rolling_sum_feature(df, 'TeamA', 11, 'ScoreA') - df.ScoreA
    df['RecentGA10A'] = rolling_sum_feature(df, 'TeamA', 11, 'ScoreB') - df.ScoreB
    df['RecentGF10B'] = rolling_sum_feature(df, 'TeamB', 11, 'ScoreB') - df.ScoreB
    df['RecentGA10B'] = rolling_sum_feature(df, 'TeamB', 11, 'ScoreA') - df.ScoreA

    df, elos = calculate_elo(df, verbose=verbose)
    df['EloDiff'] = df['EloA'] - df['EloB']

    # Save features
    os.makedirs('./data/etl/', exist_ok=True)
    df.to_csv('./data/etl/features.csv', index=False)
    with open('./data/etl/elo.json', 'w', encoding='utf-8') as f:
        json.dump(elos, f)


if __name__ == '__main__':
    process_results()
