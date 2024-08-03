import pandas as pd
import numpy as np

df_odds = pd.read_csv('./data/raw/match-odds.csv')

df_sim = pd.read_csv('./data/sim/single_sim.csv')
df_sim = df_sim[['Round', 'Date', 'TeamA', 'TeamB', 'ScoreA', 'ScoreB']]
df_sim_2 = pd.read_csv('./data/sim/knockout_simmed_matches.csv')
df_sim_2 = df_sim_2[['Round', 'Date', 'TeamA', 'TeamB', 'ScoreA', 'ScoreB']]

df = pd.merge(left=df_odds, right=df_sim, on=['Round', 'Date', 'TeamA', 'TeamB'], how='left')
df = pd.merge(left=df, right=df_sim_2, on=['Round', 'Date', 'TeamA', 'TeamB'], how='left')

df['ScoreA'] = df.apply(lambda x: x.ScoreA_x if np.isnan(x.ScoreA_y) else x.ScoreA_y, axis=1)
df['ScoreB'] = df.apply(lambda x: x.ScoreB_x if np.isnan(x.ScoreB_y) else x.ScoreB_y, axis=1)
df = df[['Round', 'Date', 'TeamA', 'TeamB', 'OddsA', 'OddsDraw', 'OddsB', 'ScoreA', 'ScoreB']]

df['Bet'] = df.apply(lambda x: 'H' if x.ScoreA > x.ScoreB else 'D' if x.ScoreA == x.ScoreB else 'A', axis=1)

res = ['H', 'A', 'H', 'H', 'D', 'A', 'A', 'A', 'H', 'A', 'H', 'H', 'H', 'D', 'D', 'H', 'D', 'D',
       'A', 'D', 'A', 'H', 'D', 'A', 'D', 'A', 'A', 'D', 'D', 'D', 'A', 'D', 'D', 'D', 'H', 'A',
       'H', 'H', 'D', 'H', 'H', 'D', 'A', 'A', 'D', 'D', 'D', 'H', 'H', 'A', 'H']

df['Result'] = res

print(df)

corr = sum(df.Bet == df.Result)
incorr = sum((df.Bet != df.Result) & (df.Result != '-'))
print('Correct:  ', corr)
print('Incorrect:', incorr)
print('Percent:  ', corr / (corr + incorr))