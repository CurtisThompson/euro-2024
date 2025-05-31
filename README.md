# Euro 2024 Supercomputer

It is now tradition during major footballing tournaments that a load of geeks will crack out their supercomputers and predict who will lift the trophy. So why don't I give it a go?

## Background

[The Euros](https://en.wikipedia.org/wiki/UEFA_European_Championship) are a quadrennial football competition containing the best international teams in Europe. The 2024 edition, hosted by Germany, contains 24 teams.

The 24 teams are divided into 6 groups of 4. Each team plays every other team in their group once, earning 3 points for a win and 1 point for a draw. The top two teams from each group qualify for the knockout rounds as well as the four best third-place teams (ranked by points and then goal difference).

The knockout matches are singular matches with the number of teams shrinking each round from 16, to 8, to 4, and then 2 teams in the final. Fixtures are matched by a predetermined bracket based on the group-finishing positions of each team.

The competition began on the 14th July 2024.

## Data

Historical match results, goalscorers, and penalty shootouts have been taken from the [Kaggle International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) dataset (downloaded 2024-06-14). The `results.csv` file contains 47267 international football results prior to the beginning of the competition - potentially all FIFA-sanctioned international matches.

Euro 2024 fixtures and groups have been manually compiled from [Wikipedia: UEFA Euro 2024](https://en.wikipedia.org/wiki/UEFA_Euro_2024) (compiled 2024-06-11).

## Pipeline

There are several stages that have been compiled in `run_pipeline.py`:

 - `load_data.py`: Load the historical results dataset and compute features for model training.

 - `build_model.py`: Build a machine learning model to predict match scores.

 - `predict_tournament.py`: Simulate the entire Euro 2024 tournament.

## Model

The model is a multi-output [XGBoost regressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor) trained on the entirety of the `results.csv` dataset prior to the beginning of the competition. 19 input features are used.

|Feature|Type|Description|
|---|---|---|
|Neutral|bool|1 if match is held in a neutral venue|
|IsHomeA|bool|1 if Team A is at home|
|IsHomeB|bool|1 if Team B is at home
|IsMajorTournament|bool|1 if the match is at the World Cup or a continental tournament|
|IsFriendly|bool|1 if the match is a friendly|
|IsEuros|bool|1 if the match is at the Euros|
|Year|int|the year the match held|
|Recent3A|int|number of points picked up by Team A in last 3 matches|
|Recent5A|int|number of points picked up by Team A in last 5 matches|
|Recent10A|int|number of points picked up by Team A in last 10 matches|
|Recent3B|int|number of points picked up by Team B in last 3 matches|
|Recent5B|int|number of points picked up by Team B in last 5 matches|
|Recent10B|int|number of points picked up by Team B in last 10 matches|
|RecentGF10A|int|goals scored by Team A in last 10 matches|
|RecentGA10A|int|goals conceded by Team A in last 10 matches|
|RecentGF10B|int|goals scored by Team B in last 10 matches|
|RecentGA10B|int|goals conceded by Team B in last 10 matches|
|EloA|float|Elo rating of Team A before match|
|EloB|float|Elo rating of Team B before match|
|EloDiff|float|Elo rating difference between Team A and Team B|

The model has a root mean squared error (RMSE) of 1.26808 when evaluated on predicted scores.
