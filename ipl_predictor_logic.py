# ipl_predictor_logic.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression

def load_raw_data(matches_path="matches.csv", deliveries_path="deliveries.csv"):
    """
    Load and merge matches + deliveries data and normalize column names.
    Returns merged deliveries dataframe with match-level fields (city, venue, winner).
    """
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    # Normalize keys if needed
    if "id" in matches.columns:
        matches = matches.rename(columns={"id": "match_id"})
    if "batter" in deliveries.columns and "batsman" not in deliveries.columns:
        deliveries = deliveries.rename(columns={"batter": "batsman"})

    # Merge a few useful match-level fields into deliveries
    full_data = deliveries.merge(
        matches[["match_id", "city", "venue", "winner"]],
        on="match_id",
        how="left"
    )

    return full_data


def get_ui_and_role_data(full_data):
    """
    Return lists for teams, venues, cities, players and a simple player role map.
    Roles: 'Batsman', 'Bowler', 'All-Rounder', 'Unknown'
    """
    teams = sorted(full_data["batting_team"].dropna().unique().tolist())
    venues = sorted(full_data["venue"].dropna().unique().tolist())
    cities = sorted(full_data["city"].dropna().unique().tolist())
    players = sorted(full_data["batsman"].dropna().unique().tolist())

    # Derive simple roles based on aggregates
    # If columns are missing, handle gracefully
    bat_counts = pd.Series(dtype=float)
    bowl_counts = pd.Series(dtype=float)
    if "batsman_runs" in full_data.columns:
        bat_counts = full_data.groupby("batsman")["batsman_runs"].sum()
    if "bowler" in full_data.columns and "is_wicket" in full_data.columns:
        bowl_counts = full_data.groupby("bowler")["is_wicket"].sum()

    player_roles = {}
    for p in players:
        bat_score = float(bat_counts.get(p, 0))
        bowl_score = float(bowl_counts.get(p, 0))
        if bat_score > 500 and bowl_score > 50:
            player_roles[p] = "All-Rounder"
        elif bat_score > 500:
            player_roles[p] = "Batsman"
        elif bowl_score > 50:
            player_roles[p] = "Bowler"
        else:
            player_roles[p] = "Unknown"

    return {
        "teams": teams,
        "venues": venues,
        "cities": cities,
        "players": players,
        "player_roles": player_roles,
    }


# ------------------------
# Batsman performance model
# ------------------------
def train_batsman_pipeline(full_data):
    """
    Trains a simple regression model that predicts average runs per batsman.
    This model uses only the 'batsman' categorical feature.
    """
    if "batsman" not in full_data.columns or "batsman_runs" not in full_data.columns:
        raise ValueError("Required columns for batsman model missing in data.")

    df = full_data.groupby("batsman")["batsman_runs"].mean().reset_index(name="avg_runs")
    X = df[["batsman"]]
    y = df["avg_runs"]

    pre = ColumnTransformer([
        ("enc", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["batsman"])
    ], remainder="drop")

    pipe = Pipeline([
        ("pre", pre),
        ("reg", LinearRegression())
    ])
    pipe.fit(X, y)
    return pipe


# ------------------------
# Bowler performance models
# ------------------------
def train_bowler_pipelines(full_data):
    """
    Trains two models for bowlers:
      - average runs conceded per delivery (regression)
      - average wicket rate per delivery (regression)
    Both use the 'bowler' categorical feature only.
    """
    if "bowler" not in full_data.columns or "total_runs" not in full_data.columns or "is_wicket" not in full_data.columns:
        raise ValueError("Required columns for bowler models missing in data.")

    bowl_runs = full_data.groupby("bowler")["total_runs"].mean().reset_index(name="avg_runs")
    bowl_wkts = full_data.groupby("bowler")["is_wicket"].mean().reset_index(name="avg_wickets")

    # Runs model
    Xr, yr = bowl_runs[["bowler"]], bowl_runs["avg_runs"]
    pre_r = ColumnTransformer([
        ("enc", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["bowler"])
    ], remainder="drop")
    runs_pipe = Pipeline([
        ("pre", pre_r),
        ("reg", LinearRegression())
    ])
    runs_pipe.fit(Xr, yr)

    # Wickets model (wicket rate per delivery)
    Xw, yw = bowl_wkts[["bowler"]], bowl_wkts["avg_wickets"]
    pre_w = ColumnTransformer([
        ("enc", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["bowler"])
    ], remainder="drop")
    wkts_pipe = Pipeline([
        ("pre", pre_w),
        ("reg", LinearRegression())
    ])
    wkts_pipe.fit(Xw, yw)

    return runs_pipe, wkts_pipe


# ------------------------
# Win prediction model
# ------------------------
def train_win_pipeline(full_data):
    """
    Trains a logistic regression model to predict the probability that the batting team
    wins in the 2nd innings given match state.
    Features used: batting_team, bowling_team, city, venue, runs_left, balls_left, wickets_left, target_runs
    Important: for deliveries, `over` and `ball` typically have `ball` in 1..6; we convert to balls bowled accordingly.
    """
    df = full_data.copy()

    # Build target: sum of first-innings runs per match -> target_runs (+1 to chase)
    targets = (
        df[df['inning'] == 1]
        .groupby('match_id')['total_runs']
        .sum()
        .rename('target_runs')
        .reset_index()
    )
    # target to win is one more run than first innings total
    targets['target_runs'] = targets['target_runs'] + 1

    # Merge target back and focus on 2nd innings rows only
    df = df.merge(targets, on='match_id', how='left')
    df = df[(df['inning'] == 2) & (df['target_runs'].notna())].copy()

    if df.empty:
        raise ValueError("No valid 2nd-innings data with targets found.")

    # Sort to accumulate current score and wickets
    df = df.sort_values(['match_id', 'over', 'ball'])

    # Calculate cumulative current score per match (total_runs is per delivery)
    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()

    # Runs left to chase
    df['runs_left'] = df['target_runs'] - df['current_score']

    # Important: deliveries commonly index overs starting at 1 and ball in 1..6.
    # Total balls bowled in match so far = (over - 1) * 6 + ball
    df['balls_bowled'] = (df['over'].fillna(0).astype(int) - 1) * 6 + df['ball'].fillna(0).astype(int)

    # For safety, if negative due to weird rows, clip
    df['balls_bowled'] = df['balls_bowled'].clip(lower=0)

    # IPL T20 total balls = 120
    df['balls_left'] = 120 - df['balls_bowled']

    # Wickets left = 10 - cumulative wickets (clip at 0)
    df['wickets_fallen'] = df.groupby('match_id')['is_wicket'].cumsum().fillna(0)
    df['wickets_left'] = (10 - df['wickets_fallen']).clip(lower=0)

    # Binary result: 1 if batting team equals match winner
    df['result'] = (df['batting_team'] == df['winner']).astype(int)

    # Keep only rows that make sense
    df = df[df['balls_left'] >= 0]

    feature_cols = [
        'batting_team', 'bowling_team', 'city', 'venue',
        'runs_left', 'balls_left', 'wickets_left', 'target_runs'
    ]

    df_features = df[feature_cols + ['result']].dropna()

    if df_features.empty:
        raise ValueError("After cleaning, no rows left for win prediction.")

    X = df_features.drop('result', axis=1)
    y = df_features['result'].astype(int)

    # Preprocessor: onehot encode the categorical team/city/venue features; pass numeric through
    categorical = ['batting_team', 'bowling_team', 'city', 'venue']
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical)
    ], remainder="passthrough")  # numeric cols left as passthrough

    # Logistic regression for probability output
    clf = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    clf.fit(X, y)
    return clf
