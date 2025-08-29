import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------
def load_raw_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    # Normalize keys
    if "id" in matches.columns:
        matches = matches.rename(columns={"id": "match_id"})
    if "batter" in deliveries.columns and "batsman" not in deliveries.columns:
        deliveries = deliveries.rename(columns={"batter": "batsman"})

    # Merge venue & city too
    full_data = deliveries.merge(
        matches[["match_id", "city", "venue", "winner"]],
        on="match_id",
        how="left"
    )
    return full_data


# -------------------------------------------------------
# UI and Role Data
# -------------------------------------------------------
def get_ui_and_role_data(full_data):
    teams = sorted(full_data["batting_team"].dropna().unique().tolist())
    venues = sorted(full_data["venue"].dropna().unique().tolist())
    cities = sorted(full_data["city"].dropna().unique().tolist())
    players = sorted(full_data["batsman"].dropna().unique().tolist())

    # Derive simple roles based on available data
    bat_counts = full_data.groupby("batsman")["batsman_runs"].sum()
    bowl_counts = full_data.groupby("bowler")["is_wicket"].sum()

    player_roles = {}
    for p in players:
        bat_score = bat_counts.get(p, 0)
        bowl_score = bowl_counts.get(p, 0)
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


# -------------------------------------------------------
# Batsman performance model (regression)
# -------------------------------------------------------
def train_batsman_pipeline(full_data):
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


# -------------------------------------------------------
# Bowler performance models (regression)
# -------------------------------------------------------
def train_bowler_pipelines(full_data):
    df = full_data.copy()
    bowl_runs = df.groupby("bowler")["total_runs"].mean().reset_index(name="avg_runs")
    bowl_wkts = df.groupby("bowler")["is_wicket"].mean().reset_index(name="avg_wickets")

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

    # Wickets model
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


# -------------------------------------------------------
# Win prediction model (classification)
# -------------------------------------------------------
def train_win_pipeline(full_data):
    df = full_data.copy()

    targets = (
        df[df['inning'] == 1]
        .groupby('match_id')['total_runs']
        .sum()
        .rename('target_runs')
        .reset_index()
    )
    targets['target_runs'] += 1

    df = df.merge(targets, on='match_id', how='left')
    df = df[(df['inning'] == 2) & (df['target_runs'].notna())].copy()

    if df.empty:
        raise ValueError("No valid 2nd-innings data with targets found.")

    df = df.sort_values(['match_id', 'over', 'ball'])
    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['runs_left'] = df['target_runs'] - df['current_score']
    df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])
    df['wickets_left'] = 10 - df.groupby('match_id')['is_wicket'].cumsum()

    df = df[df['balls_left'] >= 0]
    df['result'] = (df['batting_team'] == df['winner']).astype(int)

    final_df = df[[
        'batting_team', 'bowling_team', 'city',
        'runs_left', 'balls_left', 'wickets_left', 'target_runs', 'result'
    ]].dropna()

    if final_df.empty:
        raise ValueError("After cleaning, no rows left for win prediction.")

    X = final_df.drop('result', axis=1)
    y = final_df['result']

    pipe = Pipeline([
        ('pre', ColumnTransformer([
            ('enc', OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
             ['batting_team', 'bowling_team', 'city'])
        ], remainder='passthrough')),
        ('clf', LinearRegression())  # keep regression probability-like
    ])
    pipe.fit(X, y)
    return pipe
