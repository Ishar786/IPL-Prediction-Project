"""
app.py

Streamlit front-end that uses ipl_predictor_logic.py.
- Loads matches.csv & deliveries.csv from repo
- Tries to load models from ./models/, if missing trains them (may take a few minutes)
- Provides:
    * Player Performance Prediction (batsman & bowler)
    * Match Win Probability (2nd innings live state)
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from ipl_predictor_logic import (
    load_raw_data,
    train_and_save_models,
    load_models_from_disk,
    predict_batsman_runs,
    predict_bowler_wickets,
    predict_bowler_econ,
    predict_win_probability,
    build_player_innings_tables,
    add_recent_form
)
import time

st.set_page_config(layout="wide", page_title="IPL Contextual Predictor")

MATCHES_CSV = "matches.csv"
DELIVERIES_CSV = "deliveries.csv"
MODEL_DIR = Path("./models")

# -------------------------
# Initialize / load data & models (cached)
# -------------------------
@st.cache_data(ttl=60 * 60 * 6)
def load_data_and_ui():
    full = load_raw_data(MATCHES_CSV, DELIVERIES_CSV)
    batsman_innings, bowler_innings = build_player_innings_tables(full)
    teams = sorted(full["batting_team"].dropna().unique().tolist())
    venues = sorted(full["venue"].dropna().unique().tolist())
    cities = sorted(full["city"].dropna().unique().tolist())
    players = sorted(full["batsman"].dropna().unique().tolist())
    bowlers = sorted(full["bowler"].dropna().unique().tolist())
    return full, batsman_innings, bowler_innings, teams, venues, cities, players, bowlers


def get_or_train_models():
    # try load
    try:
        models = load_models_from_disk()
        return models, False
    except FileNotFoundError:
        # train
        st.info("Training models (first run). This may take several minutes depending on dataset size.")
        t0 = time.time()
        models = train_and_save_models(MATCHES_CSV, DELIVERIES_CSV, save_models=True)
        t1 = time.time()
        st.success(f"Models trained & saved to ./models/ (took {int(t1-t0)}s)")
        return models, True


# Load data & UI lists
try:
    full, batsman_innings, bowler_innings, teams, venues, cities, players, bowlers = load_data_and_ui()
except FileNotFoundError as e:
    st.error(f"Data files not found in project root: {e}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Load or train models
models, trained_now = get_or_train_models()
bat_pipeline = models["bat_pipeline"]
bowl_wkts_pipeline = models["bowl_wkts_pipeline"]
bowl_econ_pipeline = models["bowl_econ_pipeline"]
win_pipeline = models["win_pipeline"]

# -------------------------
# App UI
# -------------------------
st.title("🏏 IPL Contextual Predictor")
st.markdown("This app predicts player performance (runs, wickets, econ) and live win probability using venue/opponent/form context.")

app_mode = st.sidebar.selectbox("Choose Prediction Type", ["Player Performance Prediction", "Match Win Prediction"])

# -------------------------
# Player Performance Prediction
# -------------------------
if app_mode == "Player Performance Prediction":
    st.header("🔮 Player Performance Prediction (contextual)")

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_player = st.selectbox("Select Player", players)
    with col2:
        opponent_team = st.selectbox("Select Opponent Team", ["(missing)"] + teams)
    with col3:
        venue = st.selectbox("Select Venue", ["(missing)"] + venues)

    inning = st.selectbox("Innings", [1, 2])
    balls_exp = st.number_input("Expected balls (batsman balls faced / bowler balls to bowl)", min_value=1, max_value=120, value=30)

    if st.button("Predict Player Performance"):
        # Guess role by presence in bowler list and batsman list
        role = "Unknown"
        if selected_player in players and selected_player in bowlers:
            role = "All-Rounder"
        elif selected_player in players:
            role = "Batsman"
        elif selected_player in bowlers:
            role = "Bowler"

        st.subheader(f"Prediction for {selected_player} ({role})")

        # try compute recent form from innings table
        recent_form_bat = 0.0
        recent_form_bowl = 0.0
        try:
            rf_b = add_recent_form(batsman_innings, "batsman", "runs_scored", window=5)
            rf_val = rf_b[rf_b["batsman"] == selected_player]["recent_form"].tail(1)
            if not rf_val.empty:
                recent_form_bat = float(rf_val.iat[-1])
        except Exception:
            recent_form_bat = 0.0

        try:
            rf_bo = add_recent_form(bowler_innings, "bowler", "wickets", window=5)
            rf_val_b = rf_bo[rf_bo["bowler"] == selected_player]["recent_form"].tail(1)
            if not rf_val_b.empty:
                recent_form_bowl = float(rf_val_b.iat[-1])
        except Exception:
            recent_form_bowl = 0.0

        # Batsman prediction
        if role in ["Batsman", "All-Rounder"]:
            try:
                pred_runs = predict_batsman_runs(bat_pipeline, batsman=selected_player, venue=venue, opponent=opponent_team if opponent_team != "(missing)" else None, inning=inning, recent_form=recent_form_bat, balls_faced=balls_exp)
                st.metric(label="Predicted Runs (per innings)", value=f"{pred_runs:.1f}")
            except Exception as e:
                st.error(f"Batsman prediction failed: {e}")

        # Bowler prediction
        if role in ["Bowler", "All-Rounder"]:
            try:
                pred_wkts = predict_bowler_wickets(bowl_wkts_pipeline, bowler=selected_player, venue=venue, opponent=opponent_team if opponent_team != "(missing)" else None, inning=inning, recent_form=recent_form_bowl, balls_bowled=balls_exp)
                pred_econ = predict_bowler_econ(bowl_econ_pipeline, bowler=selected_player, venue=venue, opponent=opponent_team if opponent_team != "(missing)" else None, inning=inning, recent_form=recent_form_bowl, balls_bowled=balls_exp)
                st.metric(label="Predicted Wickets (per innings)", value=f"{pred_wkts:.2f}")
                st.metric(label="Predicted Economy (runs per over)", value=f"{pred_econ:.2f}")
            except Exception as e:
                st.error(f"Bowler prediction failed: {e}")

        if role == "Unknown":
            st.warning("Player role unknown in dataset. Predictions may be unreliable.")

        st.caption("Notes: Models are trained on historical deliveries + match info. Predictions are contextual (venue/opponent/recent form).")

# -------------------------
# Match Win Prediction
# -------------------------
else:
    st.header("📊 Live Match Win Probability (2nd Innings)")

    col1, col2, col3 = st.columns(3)
    with col1:
        batting_team = st.selectbox("Batting Team", teams, key="batting")
    with col2:
        bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team], key="bowling")
    with col3:
        city = st.selectbox("Host City", ["(missing)"] + cities)

    target = st.number_input("Target Score (first innings total + 1)", min_value=1, step=1, value=180)

    col4, col5, col6 = st.columns(3)
    with col4:
        score = st.number_input("Current Score", min_value=0, step=1, value=50)
    with col5:
        overs = st.text_input("Overs Completed (format O.B e.g., 8.3)", value="8.3")
    with col6:
        wickets_down = st.number_input("Wickets Down", min_value=0, max_value=10, step=1, value=2)

    venue_sel = st.selectbox("Venue", ["(missing)"] + venues)

    if st.button("Predict Win Probability"):
        if batting_team == bowling_team:
            st.error("Batting and bowling teams must be different.")
        else:
            # parse overs like 8.3 -> 8 overs and 3 balls
            try:
                if "." in overs:
                    parts = overs.split(".")
                    completed_overs = int(parts[0])
                    balls_in_over = int(parts[1])
                else:
                    completed_overs = int(float(overs))
                    balls_in_over = 0
                if balls_in_over >= 6:
                    st.warning("Balls part should be 0..5. Adjusting to 5.")
                    balls_in_over = min(balls_in_over, 5)
            except Exception:
                st.error("Invalid overs format. Use e.g., 8.3 meaning 8 overs and 3 balls.")
                st.stop()

            total_balls_bowled = completed_overs * 6 + balls_in_over
            balls_left = 120 - total_balls_bowled
            runs_left = target - score
            wickets_left = 10 - wickets_down

            # Edge cases
            if runs_left <= 0:
                st.success(f"{batting_team} has already reached the target → win probability ≈ 100%")
            elif balls_left <= 0 or wickets_left <= 0:
                st.error(f"{batting_team} has 0% win probability (no balls or wickets left)")
            else:
                try:
                    win_prob = predict_win_probability(win_pipeline, batting_team=batting_team, bowling_team=bowling_team, city=city if city != "(missing)" else None, venue=venue_sel if venue_sel != "(missing)" else None, runs_left=runs_left, balls_left=balls_left, wickets_left=wickets_left, target_runs=target)
                    win_prob = max(0.0, min(1.0, win_prob))
                    loss_prob = 1.0 - win_prob

                    st.subheader("Prediction Probabilities")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric(label=f"{batting_team} Win %", value=f"{win_prob:.0%}")
                    with c2:
                        st.metric(label=f"{bowling_team} Win %", value=f"{loss_prob:.0%}")

                    st.progress(win_prob)
                except Exception as e:
                    st.error(f"Win prediction failed: {e}")
