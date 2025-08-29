import streamlit as st
import pandas as pd
import joblib
from ipl_predictor_logic import load_raw_data, train_and_save_models, predict_match

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="IPL Match Predictor", layout="wide")

st.title("🏏 IPL Match Outcome Predictor")

# Sidebar
st.sidebar.header("Options")
page = st.sidebar.radio("Navigate", ["Train Model", "Predict Match"])

# -----------------------------
# Train Model Page
# -----------------------------
if page == "Train Model":
    st.subheader("📂 Upload Dataset to Train Model")
    uploaded_file = st.file_uploader("Upload IPL dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Data", df.head())

        if st.button("Train Model"):
            accuracy = train_and_save_models(df)
            st.success(f"Model trained successfully ✅ | Accuracy: {accuracy:.2%}")

# -----------------------------
# Prediction Page
# -----------------------------
elif page == "Predict Match":
    st.subheader("🔮 Predict Match Outcome")

    batting_team = st.selectbox("Batting Team", [
        "Chennai Super Kings", "Mumbai Indians", "Kolkata Knight Riders",
        "Royal Challengers Bangalore", "Rajasthan Royals",
        "Sunrisers Hyderabad", "Delhi Capitals", "Punjab Kings"
    ])

    bowling_team = st.selectbox("Bowling Team", [
        "Chennai Super Kings", "Mumbai Indians", "Kolkata Knight Riders",
        "Royal Challengers Bangalore", "Rajasthan Royals",
        "Sunrisers Hyderabad", "Delhi Capitals", "Punjab Kings"
    ])

    venue = st.selectbox("Venue", [
        "Eden Gardens", "Wankhede Stadium", "M. Chinnaswamy Stadium",
        "Arun Jaitley Stadium", "M. A. Chidambaram Stadium"
    ])

    runs_left = st.number_input("Runs Left", min_value=0)
    balls_left = st.number_input("Balls Left", min_value=0, max_value=120)
    wickets_left = st.slider("Wickets Left", 0, 10, 5)
    crr = st.number_input("Current Run Rate (CRR)", min_value=0.0)
    rrr = st.number_input("Required Run Rate (RRR)", min_value=0.0)

    if st.button("Predict Winner"):
        try:
            prediction, prob = predict_match(
                batting_team, bowling_team, venue,
                runs_left, balls_left, wickets_left, crr, rrr
            )
            st.success(f"🏆 Predicted Winner: **{prediction}**")
            st.info(f"Confidence: {prob:.2%}")
        except Exception as e:
            st.error(f"Error: {e}")
