import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

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

    full_data = deliveries.merge(
        matches[["match_id", "city", "winner"]],
        on="match_id",
        how="left"
    )
    return full_data


# -------------------------------------------------------
# UI and Role Data (for dropdowns etc.)
# -------------------------------------------------------
def get_ui_and_role_data(full_data):
    teams = sorted(full_data["batting_team"].dropna().unique().tolist())
    venues = sorted(full_data["venue"].dropna().unique().tolist()) if "venue" in full_data.columns else []
    cities = sorted(full_data["city"].dropna().unique().tolist())
    players = sorted(full_data["batsman"].dropna().unique().tolist())

    player_roles = {p: "Batsman" for p in players}  # placeholder

    return {
        "teams": teams,
        "venues": venues,
        "cities": cities,
        "players": players,
        "player_roles": player_roles,
    }


# -------------------------------------------------------
# Batsman performance model
# -------------------------------------------------------
def train_batsman_pipeline(full_data):
    df = full_data.copy()
    df = df.groupby("batsman")["batsman_runs"].mean().reset_index(name="avg_runs")

    X = df[["batsman"]]
    y = df["avg_runs"]

    pre = ColumnTransformer([
        ("enc", OneHotEncoder(sparse_output=False, handle_unknown="ignore")
)
    ], remainder="drop")

    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(solver="liblinear"))
    ])
    pipe.fit(X, y > y.median())  # classify above/below median scorer

    return pipe


# -------------------------------------------------------
# Bowler performance models
# -------------------------------------------------------
def train_bowler_pipelines(full_data):
    df = full_data.copy()
    bowl_runs = df.groupby("bowler")["total_runs"].mean().reset_index(name="avg_runs")
    bowl_wkts = df.groupby("bowler")["is_wicket"].mean().reset_index(name="avg_wickets")

    # Runs model
    Xr, yr = bowl_runs[["bowler"]], bowl_runs["avg_runs"]
    pre_r = ColumnTransformer([
        ("enc", OneHotEncoder(sparse_output=False, handle_unknown="ignore")
)
    ], remainder="drop")
    runs_pipe = Pipeline([
        ("pre", pre_r),
        ("clf", LogisticRegression(solver="liblinear"))
    ])
    runs_pipe.fit(Xr, yr > yr.median())

    # Wickets model
    Xw, yw = bowl_wkts[["bowler"]], bowl_wkts["avg_wickets"]
    pre_w = ColumnTransformer([
        ("enc", OneHotEncoder(sparse=False, handle_unknown="ignore"), ["bowler"])
    ], remainder="drop")
    wkts_pipe = Pipeline([
        ("pre", pre_w),
        ("clf", LogisticRegression(solver="liblinear"))
    ])
    wkts_pipe.fit(Xw, yw > yw.median())

    return runs_pipe, wkts_pipe


# -------------------------------------------------------
# Win prediction model (FIXED)
# -------------------------------------------------------
def train_win_pipeline(full_data):
    df = full_data.copy()

    # Compute target as (first-innings total + 1)
    targets = (
        df[df['inning'] == 1]
        .groupby('match_id')['total_runs']
        .sum()
        .rename('target_runs')
        .reset_index()
    )
    targets['target_runs'] = targets['target_runs'] + 1

    # Merge target onto all deliveries
    df = df.merge(targets, on='match_id', how='left')

    # Only 2nd innings with valid targets
    df = df[(df['inning'] == 2) & (df['target_runs'].notna())].copy()

    if df.empty:
        raise ValueError("No valid 2nd-innings data with targets found. Check your CSVs.")

    # Sort and engineer features
    df = df.sort_values(['match_id', 'over', 'ball'])
    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['runs_left'] = df['target_runs'] - df['current_score']
    df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])
    df['wickets_left'] = 10 - df.groupby('match_id')['is_wicket'].cumsum()

    df = df[df['balls_left'] >= 0]

    # Label: did chasing side win?
    df['result'] = (df['batting_team'] == df['winner']).astype(int)

    final_df = df[[
        'batting_team', 'bowling_team', 'city',
        'runs_left', 'balls_left', 'wickets_left', 'target_runs', 'result'
    ]].dropna()

    if final_df.empty:
        raise ValueError("After cleaning, no rows left for training win prediction.")

    X = final_df.drop('result', axis=1)
    y = final_df['result']

    pipe = Pipeline([
        ('pre', ColumnTransformer([
            ('enc', OneHotEncoder(sparse_output=False, handle_unknown="ignore")
)
        ], remainder='passthrough')),
        ('clf', LogisticRegression(solver='liblinear', max_iter=200))
    ])
    pipe.fit(X, y)

    return pipe
