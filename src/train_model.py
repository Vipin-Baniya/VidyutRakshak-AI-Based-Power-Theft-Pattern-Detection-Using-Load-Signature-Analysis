"""
train_model.py
Feature engineering, model training (RandomForest baseline), saving artifacts.
Also includes placeholders for Prophet / LSTM forecasting integration.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

DATA_PATH = "data/meters_data.csv"
ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def create_features(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    grouped = df.groupby('Meter_ID')
    features = []
    for meter_id, group in grouped:
        meter_data = {}
        meter_data['Meter_ID'] = meter_id
        meter_data['avg_consumption'] = group['Consumption_kWh'].mean()
        meter_data['std_consumption'] = group['Consumption_kWh'].std()
        meter_data['max_consumption'] = group['Consumption_kWh'].max()
        meter_data['min_consumption'] = group['Consumption_kWh'].min()
        meter_data['zero_consumption_hours'] = (group['Consumption_kWh'] == 0).sum()
        meter_data['low_consumption_hours'] = (group['Consumption_kWh'] < 0.1).sum()
        night_consumption = group[group['hour'].between(0,5)]['Consumption_kWh'].mean()
        day_consumption = group[group['hour'].between(8,20)]['Consumption_kWh'].mean()
        meter_data['night_day_ratio'] = (night_consumption / day_consumption) if day_consumption>0 else 0
        hourly_avg = group.groupby('hour')['Consumption_kWh'].mean()
        meter_data['hourly_profile_std'] = hourly_avg.std()
        weekday_avg = group[group['day_of_week']<5]['Consumption_kWh'].mean()
        weekend_avg = group[group['day_of_week']>=5]['Consumption_kWh'].mean()
        meter_data['weekend_weekday_ratio'] = (weekend_avg/weekday_avg) if weekday_avg>0 else 0
        features.append(meter_data)
    feat_df = pd.DataFrame(features).set_index('Meter_ID').fillna(0)
    return feat_df

def train(meters=50, days=30, seed=42):
    df = pd.read_csv(DATA_PATH)
    feat = create_features(df)
    labels = df.drop_duplicates('Meter_ID').set_index('Meter_ID')['Theft_Flag']
    X = feat.copy()
    y = labels.loc[X.index]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    # save artifacts
    joblib.dump(model, os.path.join(MODEL_DIR, "power_theft_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_columns.joblib"))
    # ensure MODEL_DIR exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    # confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png", bbox_inches='tight')
    print("Saved model and artifacts to", MODEL_DIR)


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="Train power theft model")
    parser.add_argument("--meters", type=int, help="Number of meters to simulate/use (overrides fast/full)", default=None)
    parser.add_argument("--days", type=int, help="Number of days to simulate/use (overrides fast/full)", default=None)
    parser.add_argument("--fast", action="store_true", help="Use fast demo settings (meters=50, days=30)")
    parser.add_argument("--full", action="store_true", help="Use full settings (meters=200, days=90)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    # decide settings
    if args.full:
        meters = 200
        days = 90
    elif args.fast:
        meters = 50
        days = 30
    else:
        # use explicit args if provided, otherwise default to fast demo
        meters = args.meters if args.meters is not None else 50
        days = args.days if args.days is not None else 30
    print(f\"Training with meters={meters}, days={days}, seed={args.seed}\")
    train(meters=meters, days=days, seed=args.seed)
