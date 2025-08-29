import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")
    return matches, deliveries

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

# ----------------------------
# Main App
# ----------------------------
def main():
    st.title("🏏 IPL Score Predictor")

    matches, deliveries = load_data()
    model = load_model()

    if model is None:
        st.error("⚠️ Model not found! Please train and save it as model.pkl")
        return

    # Sidebar inputs
    st.sidebar.header("Match Setup")
    teams = sorted(matches["team1"].unique())

    batting_team = st.sidebar.selectbox("Batting Team", teams)
    bowling_team = st.sidebar.selectbox("Bowling Team", teams)
    venue = st.sidebar.selectbox("Venue", matches["venue"].unique())

    st.sidebar.header("Match Situation")
    overs = st.sidebar.number_input("Overs Completed", min_value=5, max_value=20, value=10)
    runs = st.sidebar.number_input("Current Runs", min_value=0, value=50)
    wickets = st.sidebar.number_input("Wickets Fallen", min_value=0, max_value=10, value=2)
    runs_last_5 = st.sidebar.number_input("Runs in Last 5 Overs", min_value=0, value=30)

    if st.sidebar.button("Predict Score"):
        # Feature order should match your model training
        features = pd.DataFrame({
            "batting_team": [batting_team],
            "bowling_team": [bowling_team],
            "venue": [venue],
            "overs": [overs],
            "runs": [runs],
            "wickets": [wickets],
            "runs_last_5": [runs_last_5]
        })

        # Predict
        prediction = model.predict(features)[0]
        st.success(f"Predicted Score: {int(prediction)}")

if __name__ == "__main__":
    main()
