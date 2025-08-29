import streamlit as st
import pandas as pd
from ipl_predictor_logic import IPLPredictor

# Title
st.title("🏏 IPL Match Winner Predictor")

# Initialize predictor
predictor = IPLPredictor(dataset_path="ipl_matches.csv")
df = predictor.load_raw_data()
predictor.preprocess_data()
predictor.load_model()

# Sidebar for match input
st.sidebar.header("Match Details")

team1 = st.sidebar.selectbox("Select Team 1", df["team1"].unique())
team2 = st.sidebar.selectbox("Select Team 2", df["team2"].unique())
venue = st.sidebar.selectbox("Select Venue", df["venue"].unique())
toss_winner = st.sidebar.selectbox("Select Toss Winner", df["toss_winner"].unique())
toss_decision = st.sidebar.selectbox("Toss Decision", df["toss_decision"].unique())

# Predict button
if st.sidebar.button("Predict Winner"):
    winner = predictor.predict(team1, team2, venue, toss_winner, toss_decision)
    st.success(f"🏆 Predicted Winner: **{winner}**")
