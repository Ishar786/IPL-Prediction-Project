import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

# -----------------------------
# Load Raw Data
# -----------------------------
def load_raw_data(filepath: str):
    """
    Load IPL dataset from CSV.
    """
    df = pd.read_csv(filepath)
    return df


# -----------------------------
# Preprocessing / Feature Engineering
# -----------------------------
def preprocess_data(df: pd.DataFrame):
    """
    Clean and transform dataset for ML.
    """
    # Drop null values if any
    df = df.dropna()

    # Example features (you can expand based on dataset)
    features = [
        "batting_team", "bowling_team", "venue",
        "runs_left", "balls_left", "wickets_left", "crr", "rrr"
    ]
    target = "winner"

    # Encode categorical features
    df_encoded = pd.get_dummies(df[features], drop_first=True)
    X = df_encoded
    y = df[target]

    return X, y, df_encoded.columns


# -----------------------------
# Train and Save Model
# -----------------------------
def train_and_save_models(df: pd.DataFrame, model_dir: str = "models"):
    """
    Train RandomForest and save model + columns.
    """
    X, y, feature_cols = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {acc:.2%}")

    # Save model + columns
    Path(model_dir).mkdir(exist_ok=True)
    joblib.dump(model, f"{model_dir}/ipl_model.pkl")
    joblib.dump(feature_cols, f"{model_dir}/feature_cols.pkl")

    return acc


# -----------------------------
# Prediction
# -----------------------------
def predict_match(batting_team, bowling_team, venue,
                  runs_left, balls_left, wickets_left, crr, rrr,
                  model_path="models/ipl_model.pkl",
                  cols_path="models/feature_cols.pkl"):
    """
    Predict match outcome (which team wins).
    """
    # Load model + columns
    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)

    # Build input dataframe
    input_dict = {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "crr": crr,
        "rrr": rrr
    }
    input_df = pd.DataFrame([input_dict])

    # One-hot encode same as training
    input_encoded = pd.get_dummies(input_df).reindex(columns=feature_cols, fill_value=0)

    # Prediction
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded).max()

    return prediction, probability
