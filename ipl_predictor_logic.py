import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

class IPLPredictor:
    def __init__(self, dataset_path="ipl_matches.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self.model = None
        self.label_encoders = {}

    def load_raw_data(self):
        """Load IPL dataset from repo"""
        self.df = pd.read_csv(self.dataset_path)
        return self.df

    def preprocess_data(self):
        """Clean and encode categorical data"""
        df = self.df.copy()

        # Drop rows with missing target
        df = df.dropna(subset=["winner"])

        # Encode categorical variables
        categorical_cols = ["team1", "team2", "venue", "toss_winner", "toss_decision", "winner"]
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        self.df = df
        return df

    def train_model(self):
        """Train Random Forest model"""
        df = self.df.copy()
        X = df[["team1", "team2", "venue", "toss_winner", "toss_decision"]]
        y = df["winner"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Save model
        joblib.dump(self.model, "ipl_model.pkl")
        joblib.dump(self.label_encoders, "label_encoders.pkl")

        return self.model.score(X_test, y_test)

    def load_model(self):
        """Load saved model & encoders"""
        try:
            self.model = joblib.load("ipl_model.pkl")
            self.label_encoders = joblib.load("label_encoders.pkl")
        except:
            self.load_raw_data()
            self.preprocess_data()
            self.train_model()

    def predict(self, team1, team2, venue, toss_winner, toss_decision):
        """Predict winner"""
        if self.model is None:
            self.load_model()

        # Encode inputs
        input_data = {
            "team1": self.label_encoders["team1"].transform([team1])[0],
            "team2": self.label_encoders["team2"].transform([team2])[0],
            "venue": self.label_encoders["venue"].transform([venue])[0],
            "toss_winner": self.label_encoders["toss_winner"].transform([toss_winner])[0],
            "toss_decision": self.label_encoders["toss_decision"].transform([toss_decision])[0],
        }

        X_input = pd.DataFrame([input_data])
        winner_encoded = self.model.predict(X_input)[0]

        winner = self.label_encoders["winner"].inverse_transform([winner_encoded])[0]
        return winner
