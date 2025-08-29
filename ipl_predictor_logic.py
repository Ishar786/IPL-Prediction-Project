# ipl_predictor_logic.py
"""
Backend logic for IPL predictor:
- load_raw_data(): load and merge matches & deliveries
- build_player_innings_tables(): create per-innings records for batsmen and bowlers (with context)
- train_models(): trains and saves models (batsman_runs, bowler_wickets, bowler_econ, win_model)
- helper prediction functions for app.py
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

##### -------------------------
##### Data loading & feature engineering
##### -------------------------
def load_raw_data(matches_path="matches.csv", deliveries_path="deliveries.csv"):
    matches = pd.read_csv(matches_path, parse_dates=True, low_memory=False)
    deliveries = pd.read_csv(deliveries_path, low_memory=False)

    # Normalize
    if "id" in matches.columns:
        matches = matches.rename(columns={"id": "match_id"})
    if "batter" in deliveries.columns and "batsman" not in deliveries.columns:
        deliveries = deliveries.rename(columns={"batter": "batsman"})

    # Merge basic match info into deliveries
    if {"match_id", "city", "venue", "winner", "team1", "team2", "date"}.issubset(matches.columns):
        matches_subset = matches[["match_id", "city", "venue", "winner", "team1", "team2", "date"]]
    else:
        # fallback: take whatever columns exist
        subset_cols = [c for c in ["match_id", "city", "venue", "winner", "team1", "team2", "date"] if c in matches.columns]
        matches_subset = matches[subset_cols]

    full = deliveries.merge(matches_subset, on="match_id", how="left")
    # Ensure over and ball numeric
    if 'over' in full.columns:
        full['over'] = pd.to_numeric(full['over'], errors='coerce').fillna(0).astype(int)
    if 'ball' in full.columns:
        full['ball'] = pd.to_numeric(full['ball'], errors='coerce').fillna(0).astype(int)
    return full

def build_player_innings_tables(full):
    """
    Build:
      - batsman_innings: each row is a batsman's performance in one innings (match_id + batting_team + runs)
      - bowler_innings: each row is a bowler's performance in one innings (wickets + runs_conceded)
    Also compute 'opponent' and include venue & innings number and match date (if available)
    """
    # ensure required columns
    required = ['match_id', 'inning', 'batting_team', 'bowling_team', 'batsman', 'bowler', 'batsman_runs', 'total_runs', 'is_wicket', 'venue']
    for c in required:
        if c not in full.columns:
            raise ValueError(f"Required column '{c}' missing from deliveries data.")

    # Convert date if present in merged match info
    if 'date' in full.columns:
        try:
            full['match_date'] = pd.to_datetime(full['date'])
        except Exception:
            full['match_date'] = pd.NaT
    else:
        full['match_date'] = pd.NaT

    # Batsman innings-level aggregation
    batsman_innings = (
        full
        .groupby(['match_id', 'inning', 'batting_team', 'batsman', 'venue', 'match_date'], dropna=False)
        .agg(runs_scored=('batsman_runs', 'sum'),
             balls_faced=('ball', 'count'))  # approximate balls faced using number of deliveries where he batted
        .reset_index()
    )

    # Add opponent team (team not batting in that innings)
    # To get opponent, find unique teams in match (team1/team2 from matches could be used). We'll use bowling_team from deliveries
    # For a batsman innings, opponent is the bowling_team mode for that match+inning
    opponent_series = (
        full.groupby(['match_id', 'inning'])['bowling_team']
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else (s.iloc[0] if len(s)>0 else None))
        .reset_index()
    ).rename(columns={'bowling_team': 'opponent_team'})

    batsman_innings = batsman_innings.merge(opponent_series, on=['match_id', 'inning'], how='left')

    # Bowler innings-level aggregation
    bowler_innings = (
        full
        .groupby(['match_id', 'inning', 'bowling_team', 'bowler', 'venue', 'match_date'], dropna=False)
        .agg(runs_conceded=('total_runs', 'sum'),
             wickets=('is_wicket', 'sum'),
             balls_bowled=('ball', 'count'))
        .reset_index()
    )
    # Opponent for bowler (batting_team mode)
    opponent_b = (
        full.groupby(['match_id', 'inning'])['batting_team']
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else (s.iloc[0] if len(s)>0 else None))
        .reset_index()
    ).rename(columns={'batting_team': 'opponent_team'})

    bowler_innings = bowler_innings.merge(opponent_b, on=['match_id', 'inning'], how='left')

    # Add derived features
    batsman_innings['strike_rate'] = (batsman_innings['runs_scored'] / batsman_innings['balls_faced']).replace([np.inf, -np.inf], 0).fillna(0) * 100
    bowler_innings['econ'] = (bowler_innings['runs_conceded'] / bowler_innings['balls_bowled']).replace([np.inf, -np.inf], 0).fillna(0) * 6

    # Order by date if available for later form calculation
    batsman_innings = batsman_innings.sort_values(['match_date', 'match_id']).reset_index(drop=True)
    bowler_innings = bowler_innings.sort_values(['match_date', 'match_id']).reset_index(drop=True)

    return batsman_innings, bowler_innings

def add_recent_form(df, player_col, target_col, group_cols=None, window=5):
    """
    Add a rolling recent-form column (mean of last `window` target_col values for the player).
    The function returns a new dataframe with column 'recent_form' (player recent mean).
    """
    df = df.copy()
    df['recent_form'] = 0.0
    # Use match_date ordering if exists
    order_cols = []
    if 'match_date' in df.columns:
        order_cols = ['match_date', 'match_id']
    else:
        order_cols = ['match_id']

    df = df.sort_values(order_cols)
    # compute rolling mean per player
    df['recent_form'] = (
        df.groupby(player_col)[target_col]
          .apply(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean().fillna(0))
          .reset_index(level=0, drop=True)
    )
    return df

##### -------------------------
##### Model training / saving / loading
##### -------------------------
def train_and_save_models(matches_path="matches.csv", deliveries_path="deliveries.csv", save_models=True):
    """
    Full training pipeline:
      - loads data
      - builds batsman_innings & bowler_innings tables
      - adds recent form features
      - trains:
         * batsman_runs_model -> RandomForestRegressor
         * bowler_wickets_model -> RandomForestRegressor (predict wickets per innings)
         * bowler_econ_model -> RandomForestRegressor
         * win_model -> RandomForestClassifier (trained on per-delivery states)
      - saves models to MODEL_DIR as joblib files
    Returns dictionary of trained models and label encoders/pipelines.
    """
    full = load_raw_data(matches_path, deliveries_path)
    batsman_innings, bowler_innings = build_player_innings_tables(full)

    # Add recent form
    batsman_innings = add_recent_form(batsman_innings, player_col='batsman', target_col='runs_scored', window=5)
    bowler_innings = add_recent_form(bowler_innings, player_col='bowler', target_col='wickets', window=5)

    # Prepare training sets
    # Batsman runs per innings model
    bat_feat_cols = ['batsman', 'venue', 'opponent_team', 'inning', 'recent_form', 'balls_faced']
    X_bat = batsman_innings[bat_feat_cols].fillna('(missing)')
    y_bat = batsman_innings['runs_scored'].fillna(0)

    # Bowler wickets per innings (regression for expected wickets)
    bowl_feat_cols = ['bowler', 'venue', 'opponent_team', 'inning', 'recent_form', 'balls_bowled']
    X_bowl = bowler_innings[bowl_feat_cols].fillna('(missing)')
    y_bowl = bowler_innings['wickets'].fillna(0)

    # Bowler economy model (runs per over)
    X_bowl_econ = bowler_innings[bowl_feat_cols].fillna('(missing)')
    y_bowl_econ = bowler_innings['econ'].fillna(0)

    # Preprocessing Pipeline for players (categorical one-hot + numeric scaler)
    categorical_cols_bat = ['batsman', 'venue', 'opponent_team']
    numeric_cols_bat = ['inning', 'recent_form', 'balls_faced']

    pre_bat = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols_bat)
    ], remainder='passthrough')  # numeric columns passthrough

    bat_pipeline = Pipeline([
        ("pre", pre_bat),
        ("model", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42))
    ])

    bat_pipeline.fit(X_bat, y_bat)

    # Bowler pipelines
    categorical_cols_bowl = ['bowler', 'venue', 'opponent_team']
    pre_bowl = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols_bowl)
    ], remainder='passthrough')

    bowl_wkts_pipeline = Pipeline([
        ("pre", pre_bowl),
        ("model", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42))
    ])
    bowl_wkts_pipeline.fit(X_bowl, y_bowl)

    bowl_econ_pipeline = Pipeline([
        ("pre", pre_bowl),
        ("model", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42))
    ])
    bowl_econ_pipeline.fit(X_bowl_econ, y_bowl_econ)

    # -------- Win model: train on ball-by-ball states (like earlier but now using RandomForestClassifier)
    # Build per-delivery states for innings 2 as we did previously
    # Steps: use deliveries rows for inning 2, compute cumulative runs, balls left, wickets left, target
    dd = full.copy()
    # compute first innings totals
    targets = (
        dd[dd['inning'] == 1]
        .groupby('match_id')['total_runs']
        .sum()
        .reset_index()
        .rename(columns={'total_runs': 'target_runs'})
    )
    targets['target_runs'] = targets['target_runs'] + 1
    dd = dd.merge(targets, on='match_id', how='left')
    dd = dd[(dd['inning'] == 2) & (dd['target_runs'].notna())].copy()
    if dd.empty:
        raise ValueError("No second-innings delivery rows found to train win model.")

    dd = dd.sort_values(['match_id', 'over', 'ball'])
    dd['current_score'] = dd.groupby('match_id')['total_runs'].cumsum()
    dd['runs_left'] = dd['target_runs'] - dd['current_score']
    # calculate balls bowled using (over-1)*6 + ball
    dd['balls_bowled'] = (dd['over'].fillna(0).astype(int) - 1) * 6 + dd['ball'].fillna(0).astype(int)
    dd['balls_bowled'] = dd['balls_bowled'].clip(lower=0)
    dd['balls_left'] = 120 - dd['balls_bowled']
    dd['wickets_fallen'] = dd.groupby('match_id')['is_wicket'].cumsum().fillna(0)
    dd['wickets_left'] = (10 - dd['wickets_fallen']).clip(lower=0)
    dd['result'] = (dd['batting_team'] == dd['winner']).astype(int)

    # features and target
    win_feat_cols = ['batting_team', 'bowling_team', 'city', 'venue', 'runs_left', 'balls_left', 'wickets_left', 'target_runs']
    win_df = dd[win_feat_cols + ['result']].dropna()
    X_win = win_df[win_feat_cols]
    y_win = win_df['result']

    # Preprocessor for win model
    cat_win = ['batting_team', 'bowling_team', 'city', 'venue']
    pre_win = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_win)
    ], remainder='passthrough')

    win_pipeline = Pipeline([
        ("pre", pre_win),
        ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))
    ])
    win_pipeline.fit(X_win, y_win)

    # Evaluate quick metrics (not exhaustive)
    def eval_reg(pipe, X, y, name):
        preds = pipe.predict(X)
        rmse = mean_squared_error(y, preds, squared=False)
        r2 = r2_score(y, preds)
        print(f"{name} RMSE: {rmse:.4f}, R2: {r2:.4f}")
    def eval_clf(pipe, X, y, name):
        preds = pipe.predict(X)
        acc = accuracy_score(y, preds)
        print(f"{name} Accuracy (on training data): {acc:.4f}")

    eval_reg(bat_pipeline, X_bat, y_bat, "Batsman runs")
    eval_reg(bowl_wkts_pipeline, X_bowl, y_bowl, "Bowler wickets")
    eval_reg(bowl_econ_pipeline, X_bowl_econ, y_bowl_econ, "Bowler econ")
    eval_clf(win_pipeline, X_win, y_win, "Win model")

    models = {
        "bat_pipeline": bat_pipeline,
        "bowl_wkts_pipeline": bowl_wkts_pipeline,
        "bowl_econ_pipeline": bowl_econ_pipeline,
        "win_pipeline": win_pipeline
    }

    if save_models:
        joblib.dump(bat_pipeline, MODEL_DIR / "bat_pipeline.joblib")
        joblib.dump(bowl_wkts_pipeline, MODEL_DIR / "bowl_wkts_pipeline.joblib")
        joblib.dump(bowl_econ_pipeline, MODEL_DIR / "bowl_econ_pipeline.joblib")
        joblib.dump(win_pipeline, MODEL_DIR / "win_pipeline.joblib")
        print(f"Saved models to {MODEL_DIR}")

    return models

##### -------------------------
##### Prediction helpers (used by Streamlit app)
##### -------------------------
def load_models_from_disk():
    paths = {
        "bat_pipeline": MODEL_DIR / "bat_pipeline.joblib",
        "bowl_wkts_pipeline": MODEL_DIR / "bowl_wkts_pipeline.joblib",
        "bowl_econ_pipeline": MODEL_DIR / "bowl_econ_pipeline.joblib",
        "win_pipeline": MODEL_DIR / "win_pipeline.joblib"
    }
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Model files missing: {missing}. Run train_and_save_models() first.")

    return {k: joblib.load(v) for k, v in paths.items()}

def predict_batsman_runs(bat_pipeline, batsman, venue, opponent, inning=1, recent_form=0.0, balls_faced=30):
    x = pd.DataFrame([{
        'batsman': batsman,
        'venue': venue if venue is not None else '(missing)',
        'opponent_team': opponent if opponent is not None else '(missing)',
        'inning': inning,
        'recent_form': recent_form,
        'balls_faced': balls_faced
    }])
    pred = bat_pipeline.predict(x)[0]
    return float(max(0.0, pred))

def predict_bowler_wickets(bowl_wkts_pipeline, bowler, venue, opponent, inning=1, recent_form=0.0, balls_bowled=4*6):
    x = pd.DataFrame([{
        'bowler': bowler,
        'venue': venue if venue is not None else '(missing)',
        'opponent_team': opponent if opponent is not None else '(missing)',
        'inning': inning,
        'recent_form': recent_form,
        'balls_bowled': balls_bowled
    }])
    return float(max(0.0, bowl_wkts_pipeline.predict(x)[0]))

def predict_bowler_econ(bowl_econ_pipeline, bowler, venue, opponent, inning=1, recent_form=0.0, balls_bowled=4*6):
    x = pd.DataFrame([{
        'bowler': bowler,
        'venue': venue if venue is not None else '(missing)',
        'opponent_team': opponent if opponent is not None else '(missing)',
        'inning': inning,
        'recent_form': recent_form,
        'balls_bowled': balls_bowled
    }])
    return float(max(0.0, bowl_econ_pipeline.predict(x)[0]))

def predict_win_probability(win_pipeline, batting_team, bowling_team, city, venue, runs_left, balls_left, wickets_left, target_runs):
    x = pd.DataFrame([{
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'city': city,
        'venue': venue,
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets_left': wickets_left,
        'target_runs': target_runs
    }])
    # RandomForestClassifier -> predict_proba
    proba = win_pipeline.predict_proba(x)[0]
    # class 1 corresponds to batting-team-wins if training was done that way
    # get probability of class 1
    if win_pipeline.classes_.tolist().index(1) >= 0:
        idx = list(win_pipeline.classes_).index(1)
    else:
        # fallback
        idx = 1 if len(proba) > 1 else 0
    return float(proba[idx])



def train_and_save_models(full_data, model_dir="models"):
    """Train all models and save them into a models folder."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Train
    batsman_pipe = train_batsman_pipeline(full_data)
    bowler_runs_pipe, bowler_wickets_pipe = train_bowler_pipelines(full_data)
    win_pipe = train_win_pipeline(full_data)

    # Save
    joblib.dump(batsman_pipe, f"{model_dir}/batsman_pipe.pkl")
    joblib.dump(bowler_runs_pipe, f"{model_dir}/bowler_runs_pipe.pkl")
    joblib.dump(bowler_wickets_pipe, f"{model_dir}/bowler_wickets_pipe.pkl")
    joblib.dump(win_pipe, f"{model_dir}/win_pipe.pkl")

    print("✅ All models trained and saved successfully.")
    return batsman_pipe, bowler_runs_pipe, bowler_wickets_pipe, win_pipe
