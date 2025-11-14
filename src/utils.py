"""
utils.py
Shared helper functions for feature engineering, severity scoring, and billing estimate.
"""
import pandas as pd
import numpy as np

def create_features_from_meter_df(df):
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

def severity_score(row, prob):
    # row: Series of engineered features (avg_consumption, zero_consumption_hours, etc)
    score = 0.0
    score += min(row.get('zero_consumption_hours',0) / 24.0, 1.0) * 2.0
    score += min(row.get('night_day_ratio',0), 5.0) * 1.5
    score += (1.0 - min(row.get('hourly_profile_std',1.0), 1.0)) * 1.0
    score += prob * 2.0
    # normalize roughly to 0-10
    score = max(0.0, min(score, 10.0))
    if score < 2.5:
        return 'Low'
    elif score < 5.0:
        return 'Medium'
    elif score < 7.5:
        return 'High'
    else:
        return 'Critical'

def estimate_monthly_bill(df_meter, rate_per_kwh=6.5):
    # df_meter: hourly readings for a single meter over a month
    total_kwh = df_meter['Consumption_kWh'].sum()
    est_bill = total_kwh * rate_per_kwh
    return est_bill, total_kwh
