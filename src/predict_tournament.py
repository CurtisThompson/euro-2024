import json
import pandas as pd
import numpy as np

from match_model import MatchModel


def load_euro_games():
    # Load data
    df = pd.read_csv('./data/raw/euro-games.csv')

    # Change column names
    df.columns = ['Round', 'Date', 'TeamA', 'TeamB', 'ScoreA', 'ScoreB', 'Tournament', 'City', 'Country', 'Neutral']

    # Drop scores
    df = df.drop(['ScoreA', 'ScoreB'], axis=1)

    return df


def load_results_data():
    df_results = pd.read_csv('./data/etl/features.csv')
    with open('./data/etl/elo.json', 'r', encoding='utf-8') as f:
        elos = json.load(f)
    return df_results, elos


def load_model():
    return MatchModel().load_model('./data/models/supercomputer.model')


def add_flag_features(df):
    # Add flags
    df['Neutral'] = df['Neutral'].astype('int')
    df['IsHomeA'] = df.TeamA.eq('Germany').astype(int)
    df['IsHomeB'] = df.TeamB.eq('Germany').astype(int)
    major_tournaments = ['FIFA World Cup', 'Copa América', 'African Cup of Nations', 'AFC Asian Cup',
                        'CONCACAF Championship', 'UEFA Euro']
    df['IsMajorTournament'] = df.Tournament.isin(major_tournaments).astype('int')
    df['IsFriendly'] = df.Tournament.eq('Friendly').astype('int')
    df['IsEuros'] = df.Tournament.eq('UEFA Euro').astype('int')

    # Generate date features
    df['Year'] = pd.DatetimeIndex(df.Date).year

    return df


def simulate_round(round_num, df, df_results, elos, model):
    # Get fixtures for round
    df_round = df.loc[df.Round == round_num].copy()

    # Calculate form features for fixtures
    df_round = calculate_round_features(df_round, df_results, elos)

    # Make predictions
    columns = ['Neutral', 'IsHomeA', 'IsHomeB', 'IsMajorTournament', 'IsFriendly', 'IsEuros', 'Year',
               'Recent3A', 'Recent5A', 'Recent10A', 'Recent3B', 'Recent5B', 'Recent10B', 'EloA', 'EloB']
    df_round[['ScoreA', 'ScoreB']] = model.predict(df_round[columns])
    
    # Update elos and points
    df_round['PointsA'] = (3 * (df_round.ScoreA > df_round.ScoreB)) + (df_round.ScoreA == df_round.ScoreB)
    df_round['PointsB'] = (3 * (df_round.ScoreB > df_round.ScoreA)) + (df_round.ScoreB == df_round.ScoreA)
    elos = update_elos(df_round, elos)

    # Add results to results frame
    df_switched = df_round.copy()
    df_switched[['IsHomeA', 'IsHomeB', 'ScoreA',
                 'ScoreB', 'EloA', 'EloB',
                 'Recent3A', 'Recent5A', 'Recent10A',
                 'Recent3B', 'Recent5B', 'Recent10B',
                 'TeamA', 'TeamB', 'PointsA', 'PointsB']] = df_switched[['IsHomeB', 'IsHomeA', 'ScoreB',
                                                                         'ScoreA', 'EloB', 'EloA',
                                                                         'Recent3B', 'Recent5B', 'Recent10B',
                                                                         'Recent3A', 'Recent5A', 'Recent10A',
                                                                         'TeamB', 'TeamA', 'PointsB', 'PointsA']]
    df_results = pd.concat([df_results, df_round, df_switched], ignore_index=True)

    # Return everything
    return df_round, df_results, elos


def calculate_round_features(df, df_results, elos):
    df['Recent3A'] = df.TeamA.apply(lambda x: df_results.loc[df_results.TeamA == x, 'PointsA'].tail(3).sum())
    df['Recent3B'] = df.TeamB.apply(lambda x: df_results.loc[df_results.TeamA == x, 'PointsA'].tail(3).sum())
    df['Recent5A'] = df.TeamA.apply(lambda x: df_results.loc[df_results.TeamA == x, 'PointsA'].tail(5).sum())
    df['Recent5B'] = df.TeamB.apply(lambda x: df_results.loc[df_results.TeamA == x, 'PointsA'].tail(5).sum())
    df['Recent10A'] = df.TeamA.apply(lambda x: df_results.loc[df_results.TeamA == x, 'PointsA'].tail(10).sum())
    df['Recent10B'] = df.TeamB.apply(lambda x: df_results.loc[df_results.TeamA == x, 'PointsA'].tail(10).sum())
    df['EloA'] = df.TeamA.map(elos)
    df['EloB'] = df.TeamB.map(elos)

    return df


def update_elos(df, elos, K=32):
    for index, game in df.iterrows():
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
    
    return elos


def calculate_groups(df):
    # Import groups
    df_groups = pd.read_csv('./data/raw/euro-groups.csv')

    # Create list of results
    df_a = df[['TeamA', 'ScoreA', 'ScoreB', 'PointsA']]
    df_a.columns = ['Team', 'GF', 'GA', 'Pts']
    df_b = df[['TeamB', 'ScoreB', 'ScoreA', 'PointsB']]
    df_b.columns = ['Team', 'GF', 'GA', 'Pts']
    df_res = pd.concat([df_a, df_b])
    df_res['P'] = 1
    df_res['W'] = (df_res.GF > df_res.GA).astype(int)
    df_res['D'] = (df_res.GF == df_res.GA).astype(int)
    df_res['L'] = (df_res.GF < df_res.GA).astype(int)
    df_res['GD'] = df_res.GF - df_res.GA
    df_res = df_res.groupby('Team').sum().reset_index()[['Team', 'P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']]

    # Join results with groups
    df_groups = pd.merge(left=df_groups, right=df_res, how='inner', on='Team')

    # Sort groups
    df_groups['Pts'] = df_groups['Pts'].astype(int)
    df_groups = df_groups.sort_values(by=['Group', 'Pts', 'GD', 'GF'],
                                      ascending=[True, False, False, False],
                                      ignore_index=True)
    
    return df_groups


def get_group_positions(df):
    # Calculate group positions
    df = df.sort_values(by=['Group', 'Pts', 'GD', 'GF'],
                        ascending=[True, False, False, False],
                        ignore_index=True)
    df['GroupPos'] = df.groupby('Group').cumcount()+1
    df['RoundPos'] = df.apply(lambda x: str(x.Group) + str(x.GroupPos), axis=1)

    # Sort third place
    df_third = df.loc[df.GroupPos == 3].copy().sort_values(by=['Pts', 'GD', 'GF'],
                                                           ascending=[False, False, False],
                                                           ignore_index=True)
    df_third['ThirdPos'] = df_third.index+1
    df_third['ThirdRoundPos'] = df_third.apply(lambda x: 'X' + str(x.ThirdPos), axis=1)
    
    # Get dict of positions
    pos_dict = {**dict(zip(df.RoundPos, df.Team)),  **dict(zip(df_third.ThirdRoundPos, df_third.Team))}
    return pos_dict


def get_knockout_positions(df, tag):
    pos_dict = {}
    for index, game in df.reset_index(drop=True).iterrows():
        if game.ScoreA > game.ScoreB:
            winner = game.TeamA
        elif game.ScoreB > game.ScoreA:
            winner = game.TeamB
        elif np.random.rand() >= 0.5:
            winner = game.TeamA
        else:
            winner = game.TeamB
        pos_dict[tag+str(index+1)] = winner
    return pos_dict


def fill_in_future_fixtures(fixtures, positions):
    fixtures['TeamA'] = fixtures.TeamA.replace(positions)
    fixtures['TeamB'] = fixtures.TeamB.replace(positions)
    return fixtures


# Load future games in Euros
df = load_euro_games()
df = add_flag_features(df)

# Load ETL data
df_results, elos = load_results_data()

# Load model
model = load_model()

# Simulate tournament
# Simulate group stages
completed_results = pd.DataFrame()
for round in range(1, 4):
    round_res, df_results, elos = simulate_round(round, df, df_results, elos, model)
    completed_results = pd.concat([completed_results, round_res])
# Get final group positions
groups = calculate_groups(completed_results)
positions = get_group_positions(groups)
# Simulate knockouts
for round, tag in [(4, 'Q'), (5, 'S'), (6, 'G'), (7, 'Z')]:
    df = fill_in_future_fixtures(df, positions)
    round_res, df_results, elos = simulate_round(round, df, df_results, elos, model)
    completed_results = pd.concat([completed_results, round_res])
    positions = get_knockout_positions(round_res, tag)

print(completed_results.head(50))