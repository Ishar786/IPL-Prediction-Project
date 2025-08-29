# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

from ipl_predictor_logic import (
    load_raw_data,
    get_ui_and_role_data,
    train_and_save_models,
    load_models_from_disk,
    predict_batsman_runs,
    predict_bowler_wickets,
    predict_bowler_econ,
    predict_win_probability,
    build_player_innings_tables
)

st.set_page_config(layout="wide", page_title="IPL Predictor (Contextual)")

# --- Setup / load data & models (cached) ---
@st.cache_data(ttl=60*60*6)
def initialize(matches_path="matches.csv", deliveries_path="deliveries.csv"):
    full = load_raw_data(matches_path, deliveries_path)
    batsman_innings, bowler_innings = build_player_innings_tables(full)
    # Prepare UI lists
    teams = sorted(full['batting_team'].dropna().unique().tolist())
    venues = sorted(full['venue'].dropna().unique().tolist())
    cities = sorted(full['city'].dropna().unique().tolist())
    players = sorted(full['batsman'].dropna().unique().tolist())
    bowlers = sorted(full['bowler'].dropna().unique().tolist())
    # Attempt to load models; if missing, train
    model_dir = Path("./models")
    try:
        models = load_models_from_disk()
    except Exception:
        st.info("Training models for the first time. This may take several minutes depending on dataset size.")
        models = train_and_save_models(matches_path, deliveries_path, save_models=True)
    return full, teams, venues, cities, players, bowlers, models

try:
    full, teams, venues, cities, players, bowlers, models = initialize()
except FileNotFoundError as e:
    st.error(f"Data files missing: {e}")
    st.stop()
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

bat_pipeline = models['bat_pipeline']
bowl_wkts_pipeline = models['bowl_wkts_pipeline']
bowl_econ_pipeline = models['bowl_econ_pipeline']
win_pipeline = models['win_pipeline']

# --- App UI ---
st.title("🏏 IPL Contextual Predictor")
st.markdown("Use venue + opponent + recent form to get contextual predictions for players and a live win probability estimate.")

mode = st.sidebar.radio("Choose", ["Player Prediction", "Match Win Prediction"])

# ---------------------------
# Player Prediction
# ---------------------------
if mode == "Player Prediction":
    st.header("Player Performance Prediction (contextual)")
    col1, col2, col3 = st.columns(3)
    with col1:
        player = st.selectbox("Player (batsman or bowler)", players)
    with col2:
        role_guess = st.selectbox("Role guess (if bowler select bowler name):", ["Batsman", "Bowler", "All-Rounder"])
    with col3:
        opponent = st.selectbox("Opponent Team", ["(missing)"] + teams)

    venue = st.selectbox("Venue", ["(missing)"] + venues)
    inning = st.selectbox("Innings", [1, 2])
    recent_form = st.number_input("Recent form (avg runs or wickets last few matches) - leave 0 to auto", value=0.0, step=0.1)
    balls = st.number_input("Expected balls (batsman balls faced or bowler balls to bowl)", value=30)

    if st.button("Predict"):
        # If recent_form not provided, we try to compute from historical table (best-effort)
        if recent_form == 0.0:
            # try compute batsman's recent average runs from built innings table if present
            try:
                batsman_innings, _ = build_player_innings_tables(full)
                r = batsman_innings[batsman_innings['batsman'] == player]['runs_scored'].tail(5).mean()
                recent_form = float(r) if not pd.isna(r) else 0.0
            except Exception:
                recent_form = 0.0

        st.subheader(f"Predictions for {player} ({role_guess}) vs {opponent} at {venue}")
        # Batsman prediction
        try:
            predicted_runs = predict_batsman_runs(bat_pipeline, batsman=player, venue=venue, opponent=opponent, inning=inning, recent_form=recent_form, balls_faced=balls)
            st.metric("Predicted Runs (per innings)", f"{predicted_runs:.1f}")
        except Exception as e:
            st.error(f"Batsman prediction failed: {e}")

        # Bowler predictions (if player also bowler)
        try:
            predicted_wkts = predict_bowler_wickets(bowl_wkts_pipeline, bowler=player, venue=venue, opponent=opponent, inning=inning, recent_form=recent_form, balls_bowled=balls)
            predicted_econ = predict_bowler_econ(bowl_econ_pipeline, bowler=player, venue=venue, opponent=opponent, inning=inning, recent_form=recent_form, balls_bowled=balls)
            st.metric("Predicted Wickets (per innings)", f"{predicted_wkts:.2f}")
            st.metric("Predicted Economy (runs per over)", f"{predicted_econ:.2f}")
        except Exception as e:
            st.info("Bowler predictions unavailable or failed (player may not be a bowler in the dataset).")

        st.caption("Note: predictions are contextual — they use venue, opponent, inning and recent form. You can improve results by providing correct recent form or more refined historical data.")

# ---------------------------
# Match Win Prediction
# ---------------------------
else:
    st.header("Live Match Win Probability (2nd Innings)")
    col1, col2, col3 = st.columns(3)
    with col1:
        batting_team = st.selectbox("Batting Team", teams)
    with col2:
        bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team])
    with col3:
        city = st.selectbox("Host City", ["(missing)"] + cities)

    venue = st.selectbox("Venue", ["(missing)"] + venues)
    target = st.number_input("Target Score", min_value=1, value=180)
    score = st.number_input("Current Score", min_value=0, value=50)
    overs = st.text_input("Overs completed (format O.B e.g., 8.3)", value="8.3")
    wickets_down = st.number_input("Wickets Down", min_value=0, max_value=10, value=2)

    if st.button("Predict Win Probability"):
        # parse overs text
        try:
            if "." in overs:
                parts = overs.split(".")
                o = int(parts[0])
                b = int(parts[1])
            else:
                o = int(float(overs))
                b = 0
            if b >= 6:
                b = 5
        except Exception:
            st.error("Invalid overs format. Use e.g., 8.3 meaning 8 overs and 3 balls.")
            st.stop()

        balls_bowled = o * 6 + b
        balls_left = 120 - balls_bowled
        runs_left = target - score
        wickets_left = 10 - wickets_down

        if runs_left <= 0:
            st.success(f"{batting_team} already reached target — win probability ~100%")
        elif balls_left <= 0 or wickets_left <= 0:
            st.error(f"{batting_team} has 0% win probability (no balls or wickets left).")
        else:
            try:
                win_prob = predict_win_probability(win_pipeline, batting_team=batting_team, bowling_team=bowling_team,
                                                   city=city, venue=venue, runs_left=runs_left, balls_left=balls_left,
                                                   wickets_left=wickets_left, target_runs=target)
                win_prob = max(0.0, min(1.0, win_prob))
                st.metric(f"{batting_team} Win Probability", f"{win_prob:.0%}")
                st.metric(f"{bowling_team} Win Probability", f"{(1-win_prob):.0%}")
                st.progress(win_prob)
            except Exception as e:
                st.error(f"Win prediction failed: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("Model & data notes:\n- Models are RandomForest-based and use venue/opponent/recent form.\n- If models missing, the app will train them (may take minutes).")
