"""
ipl_predictor_logic.py

Full backend logic for the IPL predictor:
- load_raw_data(matches.csv, deliveries.csv)
- build per-innings tables and recent-form features
- train_and_save_models(...) -> trains 4 models and saves to ./models/
    - bat_pipeline (predict runs per innings)
    - bowl_wkts_pipeline (predict wickets per innings)
    - bowl_econ_pipeline (predict economy runs per over)
    - win_pipeline (predict probability batting-team wins in 2nd innings)
- load_models_from_disk()
- prediction helper functions used by the Streamlit app
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Data loading + normalization
# -------------------------
def load_raw_data(matches_path="matches.csv", deliveries_path="deliveries.csv"):
    """
    Load matches and deliveries CSV files, normalize keys and merge match-level fields
    into deliveries frame. Returns merged deliveries DataFrame.
    """
    matches = pd.read_csv(matches_path, low_memory=False)
    deliveries = pd.read_csv(deliveries_path, low_memory=False)

    # Normalize keys
    if "id" in matches.columns and "match_id" not in matches.columns:
        matches = matches.rename(columns={"id": "match_id"})
    if "batter" in deliveries.columns and "batsman" not in deliveries.columns:
        deliveries = deliveries.rename(columns={"batter": "batsman"})

    # Select a minimal subset of matches to merge (if exists)
    match_cols = [c for c in ["match_id", "city", "venue", "winner", "team1", "team2", "date"] if c in matches.columns]
    matches_subset = matches[match_cols].drop_duplicates(subset=["match_id"])

    full = deliveries.merge(matches_subset, on="match_id", how="left")

    # Ensure numeric columns
    if "over" in full.columns:
        full["over"] = pd.to_numeric(full["over"], errors="coerce").fillna(0).astype(int)
    if "ball" in full.columns:
        full["ball"] = pd.to_numeric(full["ball"], errors="coerce").fillna(0).astype(int)

    return full


# -------------------------
# Build per-innings aggregates
# -------------------------
def build_player_innings_tables(full: pd.DataFrame):
    """
    Build:
      - batsman_innings: one row per batsman's innings (runs_scored, balls_faced, venue, match_date, opponent_team)
      - bowler_innings: one row per bowler's innings (runs_conceded, wickets, balls_bowled, venue, match_date, opponent_team)
    """
    # Safety checks
    required_cols = ["match_id", "inning", "batting_team", "bowling_team", "batsman", "bowler", "batsman_runs", "total_runs", "is_wicket", "venue"]
    for c in required_cols:
        if c not in full.columns:
            raise ValueError(f"Required column '{c}' not found in deliveries data.")

    # Convert match_date if exists
    if "date" in full.columns:
        try:
            full["match_date"] = pd.to_datetime(full["date"])
        except Exception:
            full["match_date"] = pd.NaT
    else:
        full["match_date"] = pd.NaT

    # Batsman innings: group by match_id, inning, batsman
    batsman_innings = (
        full
        .groupby(["match_id", "inning", "batting_team", "batsman", "venue", "match_date"], dropna=False)
        .agg(runs_scored=("batsman_runs", "sum"),
             balls_faced=("batsman_runs", lambda s: s.count()))
        .reset_index()
    )

    # Opponent for batsman innings: mode of bowling_team in that match+inning
    opponent_series = (
        full.groupby(["match_id", "inning"])["bowling_team"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else (s.iloc[0] if len(s) > 0 else None))
        .reset_index()
        .rename(columns={"bowling_team": "opponent_team"})
    )
    batsman_innings = batsman_innings.merge(opponent_series, on=["match_id", "inning"], how="left")

    # Bowler innings: group by match_id, inning, bowler
    bowler_innings = (
        full
        .groupby(["match_id", "inning", "bowling_team", "bowler", "venue", "match_date"], dropna=False)
        .agg(runs_conceded=("total_runs", "sum"),
             wickets=("is_wicket", "sum"),
             balls_bowled=("total_runs", lambda s: s.count()))
        .reset_index()
    )

    opponent_b = (
        full.groupby(["match_id", "inning"])["batting_team"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else (s.iloc[0] if len(s) > 0 else None))
        .reset_index()
        .rename(columns={"batting_team": "opponent_team"})
    )
    bowler_innings = bowler_innings.merge(opponent_b, on=["match_id", "inning"], how="left")

    # Derived metrics
    batsman_innings["strike_rate"] = (batsman_innings["runs_scored"] / batsman_innings["balls_faced"]).replace([np.inf, -np.inf], 0).fillna(0) * 100
    bowler_innings["econ"] = (bowler_innings["runs_conceded"] / bowler_innings["balls_bowled"]).replace([np.inf, -np.inf], 0).fillna(0) * 6

    # Sort by date for form
    batsman_innings = batsman_innings.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    bowler_innings = bowler_innings.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    return batsman_innings, bowler_innings


# -------------------------
# Recent form feature
# -------------------------
def add_recent_form(df: pd.DataFrame, player_col: str, target_col: str, window: int = 5):
    """
    Add 'recent_form' column: rolling mean of last `window` values of `target_col` for each player.
    The recent_form is shifted so it's based on previous matches only (no leakage).
    """
    df = df.copy()
    order_cols = ["match_date", "match_id"] if "match_date" in df.columns else ["match_id"]
    df = df.sort_values(order_cols)
    # compute shifted rolling mean per player
    df["recent_form"] = (
        df.groupby(player_col)[target_col]
          .apply(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean().fillna(0))
          .reset_index(level=0, drop=True)
    )
    return df


# -------------------------
# Train & save models
# -------------------------
def train_and_save_models(matches_path="matches.csv", deliveries_path="deliveries.csv", save_models=True):
    """
    Full training pipeline. Trains and returns dict of models.
    Models saved to ./models/ when save_models=True.
    Returns: dict with {bat_pipeline, bowl_wkts_pipeline, bowl_econ_pipeline, win_pipeline}
    """
    full = load_raw_data(matches_path, deliveries_path)
    batsman_innings, bowler_innings = build_player_innings_tables(full)

    # Add recent form features
    # For batsman: recent mean runs_scored
    batsman_innings = add_recent_form(batsman_innings, player_col="batsman", target_col="runs_scored", window=5)
    # For bowler: recent mean wickets
    bowler_innings = add_recent_form(bowler_innings, player_col="bowler", target_col="wickets", window=5)

    # ------- Prepare training data for batsman runs model -------
    bat_feat_cols = ["batsman", "venue", "opponent_team", "inning", "recent_form", "balls_faced"]
    X_bat = batsman_innings[bat_feat_cols].fillna("(missing)")
    y_bat = batsman_innings["runs_scored"].fillna(0)

    pre_bat = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["batsman", "venue", "opponent_team"])
    ], remainder="passthrough")

    bat_pipeline = Pipeline([
        ("pre", pre_bat),
        ("reg", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42))
    ])
    bat_pipeline.fit(X_bat, y_bat)

    # ------- Bowler wickets & econ models -------
    bowl_feat_cols = ["bowler", "venue", "opponent_team", "inning", "recent_form", "balls_bowled"]
    X_bowl = bowler_innings[bowl_feat_cols].fillna("(missing)")
    y_bowl_wkts = bowler_innings["wickets"].fillna(0)
    y_bowl_econ = bowler_innings["econ"].fillna(0)

    pre_bowl = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["bowler", "venue", "opponent_team"])
    ], remainder="passthrough")

    bowl_wkts_pipeline = Pipeline([
        ("pre", pre_bowl),
        ("reg", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42))
    ])
    bowl_wkts_pipeline.fit(X_bowl, y_bowl_wkts)

    bowl_econ_pipeline = Pipeline([
        ("pre", pre_bowl),
        ("reg", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42))
    ])
    bowl_econ_pipeline.fit(X_bowl, y_bowl_econ)

    # ------- Win model: build per-delivery states for 2nd innings -------
    dd = full.copy()
    # compute first-innings totals and target runs (first innings total + 1)
    targets = (
        dd[dd["inning"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "target_runs"})
    )
    targets["target_runs"] = targets["target_runs"] + 1
    dd = dd.merge(targets, on="match_id", how="left")
    dd = dd[(dd["inning"] == 2) & (dd["target_runs"].notna())].copy()
    if dd.empty:
        raise ValueError("No second-innings data found for win model training.")

    dd = dd.sort_values(["match_id", "over", "ball"])
    dd["current_score"] = dd.groupby("match_id")["total_runs"].cumsum()
    dd["runs_left"] = dd["target_runs"] - dd["current_score"]

    # balls bowled so far in T20: (over - 1) * 6 + ball
    dd["balls_bowled"] = (dd["over"].fillna(0).astype(int) - 1) * 6 + dd["ball"].fillna(0).astype(int)
    dd["balls_bowled"] = dd["balls_bowled"].clip(lower=0)
    dd["balls_left"] = 120 - dd["balls_bowled"]

    dd["wickets_fallen"] = dd.groupby("match_id")["is_wicket"].cumsum().fillna(0)
    dd["wickets_left"] = (10 - dd["wickets_fallen"]).clip(lower=0)

    dd["result"] = (dd["batting_team"] == dd["winner"]).astype(int)

    # Filter sensible rows
    dd = dd[dd["balls_left"] >= 0]
    win_feat_cols = ["batting_team", "bowling_team", "city", "venue", "runs_left", "balls_left", "wickets_left", "target_runs"]
    win_df = dd[win_feat_cols + ["result"]].dropna()

    X_win = win_df[win_feat_cols]
    y_win = win_df["result"].astype(int)

    pre_win = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
         ["batting_team", "bowling_team", "city", "venue"])
    ], remainder="passthrough")

    win_pipeline = Pipeline([
        ("pre", pre_win),
        ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))
    ])
    win_pipeline.fit(X_win, y_win)

    # -------- Save models --------
    models = {
        "bat_pipeline": bat_pipeline,
        "bowl_wkts_pipeline": bowl_wkts_pipeline,
        "bowl_econ_pipeline": bowl_econ_pipeline,
        "win_pipeline": win_pipeline,
    }

    if save_models:
        joblib.dump(bat_pipeline, MODEL_DIR / "bat_pipeline.joblib")
        joblib.dump(bowl_wkts_pipeline, MODEL_DIR / "bowl_wkts_pipeline.joblib")
        joblib.dump(bowl_econ_pipeline, MODEL_DIR / "bowl_econ_pipeline.joblib")
        joblib.dump(win_pipeline, MODEL_DIR / "win_pipeline.joblib")

    # Quick training-evaluation prints (on training data) to give feedback
    def eval_reg(pipe, X, y, name):
        preds = pipe.predict(X)
        rmse = mean_squared_error(y, preds, squared=False)
        r2 = r2_score(y, preds)
        print(f"{name} RMSE: {rmse:.4f}, R2: {r2:.4f}")

    def eval_clf(pipe, X, y, name):
        preds = pipe.predict(X)
        acc = accuracy_score(y, preds)
        print(f"{name} Accuracy (training data): {acc:.4f}")

    try:
        eval_reg(bat_pipeline, X_bat, y_bat, "Batsman runs")
        eval_reg(bowl_wkts_pipeline, X_bowl, y_bowl_wkts, "Bowler wickets")
        eval_reg(bowl_econ_pipeline, X_bowl, y_bowl_econ, "Bowler econ")
        eval_clf(win_pipeline, X_win, y_win, "Win model")
    except Exception:
        pass

    return models


# -------------------------
# Load models from disk
# -------------------------
def load_models_from_disk():
    """
    Load the 4 pipelines from ./models/ if present.
    Raises FileNotFoundError if any missing.
    """
    required = {
        "bat_pipeline": MODEL_DIR / "bat_pipeline.joblib",
        "bowl_wkts_pipeline": MODEL_DIR / "bowl_wkts_pipeline.joblib",
        "bowl_econ_pipeline": MODEL_DIR / "bowl_econ_pipeline.joblib",
        "win_pipeline": MODEL_DIR / "win_pipeline.joblib",
    }
    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}. Train using train_and_save_models().")

    return {
        "bat_pipeline": joblib.load(required["bat_pipeline"]),
        "bowl_wkts_pipeline": joblib.load(required["bowl_wkts_pipeline"]),
        "bowl_econ_pipeline": joblib.load(required["bowl_econ_pipeline"]),
        "win_pipeline": joblib.load(required["win_pipeline"]),
    }


# -------------------------
# Prediction helpers
# -------------------------
def predict_batsman_runs(bat_pipeline, batsman, venue, opponent, inning=1, recent_form=0.0, balls_faced=30):
    x = pd.DataFrame([{
        "batsman": batsman,
        "venue": venue if venue is not None else "(missing)",
        "opponent_team": opponent if opponent is not None else "(missing)",
        "inning": inning,
        "recent_form": recent_form,
        "balls_faced": balls_faced
    }])
    pred = bat_pipeline.predict(x)[0]
    return float(max(0.0, pred))


def predict_bowler_wickets(bowl_wkts_pipeline, bowler, venue, opponent, inning=1, recent_form=0.0, balls_bowled=24):
    x = pd.DataFrame([{
        "bowler": bowler,
        "venue": venue if venue is not None else "(missing)",
        "opponent_team": opponent if opponent is not None else "(missing)",
        "inning": inning,
        "recent_form": recent_form,
        "balls_bowled": balls_bowled
    }])
    pred = bowl_wkts_pipeline.predict(x)[0]
    return float(max(0.0, pred))


def predict_bowler_econ(bowl_econ_pipeline, bowler, venue, opponent, inning=1, recent_form=0.0, balls_bowled=24):
    x = pd.DataFrame([{
        "bowler": bowler,
        "venue": venue if venue is not None else "(missing)",
        "opponent_team": opponent if opponent is not None else "(missing)",
        "inning": inning,
        "recent_form": recent_form,
        "balls_bowled": balls_bowled
    }])
    pred = bowl_econ_pipeline.predict(x)[0]
    return float(max(0.0, pred))


def predict_win_probability(win_pipeline, batting_team, bowling_team, city, venue, runs_left, balls_left, wickets_left, target_runs):
    x = pd.DataFrame([{
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "city": city,
        "venue": venue,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "target_runs": target_runs
    }])
    proba = win_pipeline.predict_proba(x)[0]
    # find index of class "1" (batting-team-wins)
    classes = list(win_pipeline.named_steps["clf"].classes_) if "clf" in win_pipeline.named_steps else list(win_pipeline.classes_)
    if 1 in classes:
        idx = classes.index(1)
    else:
        idx = 1 if len(proba) > 1 else 0
    return float(proba[idx])
