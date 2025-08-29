import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import os

def load_raw_data():
    """Loads, merges, and performs initial cleaning of the raw CSV files."""
    if not os.path.exists('matches.csv') or not os.path.exists('deliveries.csv'):
        raise FileNotFoundError("Ensure 'matches.csv' and 'deliveries.csv' are in the project folder.")
        
    matches = pd.read_csv('matches.csv')
    deliveries = pd.read_csv('deliveries.csv')
    
    # Standardize column names for merging and consistency
    matches.rename(columns={'id': 'match_id'}, inplace=True)
    if 'batter' in deliveries.columns:
        deliveries.rename(columns={'batter': 'batsman'}, inplace=True)

    full_data = pd.merge(deliveries, matches, on='match_id', how='inner')
    full_data['is_wicket'] = full_data['player_dismissed'].apply(lambda x: 1 if pd.notna(x) else 0)
    return full_data

def get_ui_and_role_data(full_data):
    """Determines player roles and creates structured data for the UI dropdowns."""
    batsman_runs = full_data.groupby('batsman')['batsman_runs'].sum()
    balls_faced = full_data['batsman'].value_counts()
    bowler_wickets = full_data[full_data['is_wicket'] == 1].groupby('bowler')['is_wicket'].sum()
    balls_bowled = full_data['bowler'].value_counts()

    player_roles = {}
    all_players = set(batsman_runs.index) | set(bowler_wickets.index)

    for player in all_players:
        # Define criteria for player roles based on historical data
        is_batsman = balls_faced.get(player, 0) > 100 and batsman_runs.get(player, 0) > 500
        is_bowler = balls_bowled.get(player, 0) > 100 and bowler_wickets.get(player, 0) > 25
        if is_batsman and is_bowler:
            player_roles[player] = 'All-Rounder'
        elif is_batsman:
            player_roles[player] = 'Batsman'
        elif is_bowler:
            player_roles[player] = 'Bowler'
        else:
            player_roles[player] = 'Unknown'
            
    ui_data = {
        'teams': sorted([team for team in full_data['team1'].unique() if pd.notna(team)]),
        'venues': sorted([venue for venue in full_data['venue'].unique() if pd.notna(venue)]),
        'cities': sorted([city for city in full_data['city'].dropna().unique() if pd.notna(city)]),
        'players': sorted([p for p in all_players if p]),  # Filter out any potential None/NaN players
        'player_roles': player_roles
    }
    return ui_data

def train_batsman_pipeline(full_data):
    """Trains and returns the batsman score prediction pipeline."""
    batsman_df = full_data.groupby(['match_id', 'batsman', 'bowling_team', 'venue'])['batsman_runs'].sum().reset_index()
    X = batsman_df[['batsman', 'bowling_team', 'venue']]
    y = batsman_df['batsman_runs']
    
    pipe = Pipeline(steps=[
        ('preprocessor', ColumnTransformer([('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['batsman', 'bowling_team', 'venue'])])),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    pipe.fit(X, y)
    return pipe

def train_bowler_pipelines(full_data):
    """Trains and returns pipelines for bowler runs conceded and wickets taken."""
    bowler_df = full_data.groupby(['match_id', 'bowler', 'batting_team', 'venue']).agg(
        runs_conceded=('total_runs', 'sum'),
        wickets_taken=('is_wicket', 'sum')
    ).reset_index()
    
    X = bowler_df[['bowler', 'batting_team', 'venue']]
    
    # Runs Conceded Pipeline
    y_runs = bowler_df['runs_conceded']
    runs_pipe = Pipeline(steps=[
        ('preprocessor', ColumnTransformer([('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['bowler', 'batting_team', 'venue'])])),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    runs_pipe.fit(X, y_runs)
    
    # Wickets Taken Pipeline
    y_wickets = bowler_df['wickets_taken']
    wickets_pipe = Pipeline(steps=[
        ('preprocessor', ColumnTransformer([('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['bowler', 'batting_team', 'venue'])])),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    wickets_pipe.fit(X, y_wickets)
    
    return runs_pipe, wickets_pipe

def train_win_pipeline(full_data):
    """Trains and returns the match win prediction pipeline for the 2nd innings."""
    win_df = full_data.copy()

    # Compute target runs from first innings only
    match_targets = (
        win_df[win_df['inning'] == 1]
        .groupby('match_id')['total_runs']
        .sum()
        .reset_index()
        .rename(columns={'total_runs': 'target_runs'})
    )
    match_targets['target_runs'] += 1  # target = runs + 1

    # Merge with left join so we don't lose 2nd innings data
    win_df = win_df.merge(match_targets, on='match_id', how='left')

    # Only keep second innings rows that actually have a target
    win_df = win_df[(win_df['inning'] == 2) & (win_df['target_runs'].notna())].copy()

    if win_df.empty:
        raise ValueError("No valid 2nd innings data with targets found. Check your CSV files.")

    # Features
    win_df['current_score'] = win_df.groupby('match_id')['total_runs'].cumsum()
    win_df['runs_left'] = win_df['target_runs'] - win_df['current_score']
    win_df['balls_left'] = 120 - (win_df['over'] * 6 + win_df['ball'])
    win_df['wickets_left'] = 10 - win_df.groupby('match_id')['is_wicket'].cumsum()
    win_df = win_df[win_df['balls_left'] >= 0]

    # Label: 1 if batting team won, else 0
    win_df['result'] = (win_df['batting_team'] == win_df['winner']).astype(int)

    # Final dataset
    final_df = win_df[['batting_team', 'bowling_team', 'city',
                       'runs_left', 'balls_left', 'wickets_left',
                       'target_runs', 'result']].dropna()

    if final_df.empty:
        raise ValueError("After cleaning, no rows left for training win prediction.")

    X = final_df.drop('result', axis=1)
    y = final_df['result']

    # Pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', ColumnTransformer([
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
             ['batting_team', 'bowling_team', 'city'])
        ])),
        ('classifier', LogisticRegression(solver='liblinear'))
    ])
    pipe.fit(X, y)
    return pipe
