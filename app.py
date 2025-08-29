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

# Set wide layout
st.set_page_config(layout="wide")

@st.cache_data
def load_and_train_models():
    """Load data and train all models (cached for speed)."""
    print("CACHE MISS: Loading data and training models...")
    raw_data = load_raw_data()
    ui_data = get_ui_and_role_data(raw_data)
    batsman_pipe = train_batsman_pipeline(raw_data)
    bowler_runs_pipe, bowler_wickets_pipe = train_bowler_pipelines(raw_data)
    win_pipe = train_win_pipeline(raw_data)
    print("Models trained and cached successfully.")
    return ui_data, batsman_pipe, bowler_runs_pipe, bowler_wickets_pipe, win_pipe


# --- Load Models and UI Data ---
try:
    with st.spinner('Setting up the app for the first time... This might take a moment.'):
        ui_data, batsman_pipe, bowler_runs_pipe, bowler_wickets_pipe, win_pipe = load_and_train_models()

    teams = ui_data['teams']
    venues = ui_data['venues']
    cities = ui_data['cities']
    players = ui_data['players']
    player_roles = ui_data['player_roles']

except FileNotFoundError as e:
    st.error(f"ERROR: Data file not found. {e}")
    st.info("Please make sure 'matches.csv' and 'deliveries.csv' are in the same folder as this app.")
    st.stop()


# --- Main App Layout ---
st.title('🏏 IPL Performance and Win Predictor')
st.markdown("---")

# --- Sidebar for App Navigation ---
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
        opponent_team = st.selectbox('Select Opponent Team', [t for t in teams if t])
    with col3:
        venue = st.selectbox('Select Venue', venues)

    if st.button('Predict Player Performance'):
        role = player_roles.get(selected_player, 'Unknown')
        st.subheader(f'Prediction for {selected_player} ({role})')

        # Batsman predictions
        if role in ['Batsman', 'All-Rounder']:
            input_df_bat = pd.DataFrame({'batsman': [selected_player]})
            predicted_runs = batsman_pipe.predict(input_df_bat)[0]
            st.metric(label="Predicted Runs", value=f"~ {predicted_runs:.1f} runs")

        # Bowler predictions
        if role in ['Bowler', 'All-Rounder']:
            input_df_bowl = pd.DataFrame({'bowler': [selected_player]})
            predicted_runs = bowler_runs_pipe.predict(input_df_bowl)[0]
            predicted_wickets = bowler_wickets_pipe.predict(input_df_bowl)[0]
            st.metric(label="Predicted Wickets", value=f"~ {predicted_wickets:.1f}")
            st.metric(label="Predicted Runs Conceded", value=f"~ {predicted_runs:.1f}")

        if role == 'Unknown':
            st.warning("Player has limited historical data for a defined role. Predictions may be less accurate.")


# =========================================================
# Match Win Prediction
# =========================================================
elif app_mode == 'Match Win Prediction':
    st.header('📊 Live Match Win Probability (2nd Innings)')

    col1, col2, col3 = st.columns(3)
    with col1:
        batting_team = st.selectbox('Batting Team', teams, key='batting')
    with col2:
        bowling_team = st.selectbox('Bowling Team', teams, key='bowling')
    with col3:
        city = st.selectbox('Host City', cities)

    target = st.number_input('Target Score', min_value=1, step=1, value=180)

    col4, col5, col6 = st.columns(3)
    with col4:
        score = st.number_input('Current Score', min_value=0, step=1)
    with col5:
        overs = st.number_input('Overs Completed (e.g., 8.3)', min_value=0.0, max_value=19.5, step=0.1, format="%.1f")
    with col6:
        wickets = st.number_input('Wickets Down', min_value=0, max_value=10, step=1)

    if st.button('Predict Win Probability'):
        if batting_team == bowling_team:
            st.error("Batting and Bowling teams must be different.")
        else:
            runs_left = target - score
            balls_left = 120 - (int(overs) * 6 + round((overs % 1) * 10))
            wickets_left = 10 - wickets

            # Handle edge cases
            if runs_left <= 0:
                st.success(f"{batting_team} has a 100% win probability.")
            elif balls_left <= 0 or wickets_left <= 0:
                st.error(f"{batting_team} has a 0% win probability.")
            else:
                input_df = pd.DataFrame({
                    'batting_team': [batting_team],
                    'bowling_team': [bowling_team],
                    'city': [city],
                    'runs_left': [runs_left],
                    'balls_left': [balls_left],
                    'wickets_left': [wickets_left],
                    'target_runs': [target]
                })

                # Regression output → clamp between 0 and 1
                win_prob = win_pipe.predict(input_df)[0]
                win_prob = max(0, min(1, win_prob))
                loss_prob = 1 - win_prob

                st.subheader('Prediction Probabilities')
                col7, col8 = st.columns(2)
                with col7:
                    st.metric(label=f"{batting_team} Win %", value=f"{win_prob:.0%}")
                with col8:
                    st.metric(label=f"{bowling_team} Win %", value=f"{loss_prob:.0%}")

                st.progress(win_prob)
