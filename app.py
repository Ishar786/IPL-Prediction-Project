# app.py
import streamlit as st
import pandas as pd
# Import backend logic
from ipl_predictor_logic import (
    load_raw_data,
    get_ui_and_role_data,
    train_batsman_pipeline,
    train_bowler_pipelines,
    train_win_pipeline
)

st.set_page_config(layout="wide", page_title="IPL Predictor")

@st.cache_data(ttl=60*60*6)  # cache models for 6 hours by default
def load_and_train_models():
    """Load data and train all models (cached for speed)."""
    raw_data = load_raw_data()
    ui_data = get_ui_and_role_data(raw_data)
    batsman_pipe = train_batsman_pipeline(raw_data)
    bowler_runs_pipe, bowler_wickets_pipe = train_bowler_pipelines(raw_data)
    win_pipe = train_win_pipeline(raw_data)
    return ui_data, batsman_pipe, bowler_runs_pipe, bowler_wickets_pipe, win_pipe

# --- Load Models and UI Data ---
try:
    with st.spinner('Setting up the app (loading data and training models)...'):
        ui_data, batsman_pipe, bowler_runs_pipe, bowler_wickets_pipe, win_pipe = load_and_train_models()

    teams = ui_data['teams']
    venues = ui_data['venues']
    cities = ui_data['cities']
    players = ui_data['players']
    player_roles = ui_data['player_roles']

except FileNotFoundError as e:
    st.error(f"ERROR: Data file not found. {e}")
    st.info("Please ensure 'matches.csv' and 'deliveries.csv' are in the same folder as this app.")
    st.stop()
except Exception as e:
    st.error(f"Failed to initialize app: {e}")
    st.stop()


# --- Main App Layout ---
st.title('🏏 IPL Performance and Win Predictor')
st.markdown("---")

app_mode = st.sidebar.selectbox('Choose Prediction Type',
    ['Player Performance Prediction', 'Match Win Prediction'])


# =========================================================
# Player Performance Prediction
# =========================================================
if app_mode == 'Player Performance Prediction':
    st.header('🔮 Player Performance Prediction')

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_player = st.selectbox('Select a Player', players)
    with col2:
        opponent_team = st.selectbox('Select Opponent Team', ['(not used by model)'] + teams)
    with col3:
        venue = st.selectbox('Select Venue', ['(not used by model)'] + venues)

    if st.button('Predict Player Performance'):
        role = player_roles.get(selected_player, 'Unknown')
        st.subheader(f'Prediction for {selected_player} ({role})')

        # Batsman predictions
        if role in ['Batsman', 'All-Rounder']:
            input_df_bat = pd.DataFrame({'batsman': [selected_player]})
            try:
                predicted_runs = batsman_pipe.predict(input_df_bat)[0]
                st.metric(label="Predicted Runs", value=f"~ {predicted_runs:.1f} runs")
            except Exception as e:
                st.error(f"Could not predict batsman runs: {e}")

        # Bowler predictions
        if role in ['Bowler', 'All-Rounder']:
            input_df_bowl = pd.DataFrame({'bowler': [selected_player]})
            try:
                predicted_runs = bowler_runs_pipe.predict(input_df_bowl)[0]
                predicted_wickets = bowler_wickets_pipe.predict(input_df_bowl)[0]
                st.metric(label="Predicted Wickets", value=f"~ {predicted_wickets:.2f}")
                st.metric(label="Predicted Runs Conceded (per delivery avg)", value=f"~ {predicted_runs:.2f}")
            except Exception as e:
                st.error(f"Could not predict bowler stats: {e}")

        if role == 'Unknown':
            st.warning("Player has limited historical data for a defined role. Predictions may be less accurate.")

        st.caption("Note: Current player models are simple and use player name only. To include opponent/venue context you'd need to retrain models with those features.")


# =========================================================
# Match Win Prediction
# =========================================================
elif app_mode == 'Match Win Prediction':
    st.header('📊 Live Match Win Probability (2nd Innings)')

    col1, col2, col3 = st.columns(3)
    with col1:
        batting_team = st.selectbox('Batting Team', teams, key='batting')
    with col2:
        bowling_team = st.selectbox('Bowling Team', [t for t in teams if t != batting_team], key='bowling')
    with col3:
        city = st.selectbox('Host City', cities)

    target = st.number_input('Target Score', min_value=1, step=1, value=180)

    col4, col5, col6 = st.columns(3)
    with col4:
        score = st.number_input('Current Score', min_value=0, step=1, value=50)
    with col5:
        overs = st.number_input('Overs Completed (format: O.B e.g., 8.3)', min_value=0.0, max_value=19.5, step=0.1, format="%.1f")
    with col6:
        wickets = st.number_input('Wickets Down', min_value=0, max_value=10, step=1)

    venue = st.selectbox('Venue', venues)

    if st.button('Predict Win Probability'):
        if batting_team == bowling_team:
            st.error("Batting and Bowling teams must be different.")
        else:
            # parse overs input (e.g., 8.3 -> 8 overs and 3 balls)
            completed_overs = int(overs)
            balls_in_current_over = int(round((overs - completed_overs) * 10))
            if balls_in_current_over >= 6:
                st.warning("Ball part of overs should be 0..5 (e.g., 8.3 means 8 overs + 3 balls). Adjusting to 5 max.")
                balls_in_current_over = min(balls_in_current_over, 5)

            total_balls_bowled = completed_overs * 6 + balls_in_current_over
            balls_left = 120 - total_balls_bowled
            runs_left = target - score
            wickets_left = 10 - wickets

            # Edge cases
            if runs_left <= 0:
                st.success(f"{batting_team} has a 100% win probability (target already reached).")
            elif balls_left <= 0 or wickets_left <= 0:
                st.error(f"{batting_team} has a 0% win probability (no balls or no wickets left).")
            else:
                input_df = pd.DataFrame({
                    'batting_team': [batting_team],
                    'bowling_team': [bowling_team],
                    'city': [city],
                    'venue': [venue],
                    'runs_left': [runs_left],
                    'balls_left': [balls_left],
                    'wickets_left': [wickets_left],
                    'target_runs': [target]
                })

                try:
                    # win_pipe is a logistic regression pipeline -> use predict_proba to get probability for class 1
                    probs = win_pipe.predict_proba(input_df)
                    # class 1 is batting-team-wins
                    win_prob = float(probs[0][1])
                    win_prob = max(0.0, min(1.0, win_prob))
                    loss_prob = 1.0 - win_prob

                    st.subheader('Prediction Probabilities')
                    col7, col8 = st.columns(2)
                    with col7:
                        st.metric(label=f"{batting_team} Win %", value=f"{win_prob:.0%}")
                    with col8:
                        st.metric(label=f"{bowling_team} Win %", value=f"{loss_prob:.0%}")

                    st.progress(win_prob)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
