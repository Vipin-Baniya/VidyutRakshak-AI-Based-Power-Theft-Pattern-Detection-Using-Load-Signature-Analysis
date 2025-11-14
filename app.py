"""
app.py - Streamlit dashboard for Power Theft Detection (prototype)
Features:
 - Upload meter CSV or use sample data
 - Forecast expected load per meter (Prophet if installed, otherwise simple rolling mean)
 - Anomaly timeline visualization
 - Auto-refresh dashboard (every 5s) for demo (uses st.experimental_rerun with timer)
 - GIS Map view with folium (if available)
 - Multi-model comparison toggle (RandomForest, LogisticRegression, SVC)
 - SHAP explanation (if SHAP is installed)
 - Severity scoring and monthly bill estimator
"""
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Grid Theft Detection", layout="wide")

# Custom CSS for improved UI
st.markdown(
    """
    <style>
    .big-title { font-size:32px; font-weight:700; color:#0f4c81; }
    .subtle { color: #6c6c6c; }
    .card { background: linear-gradient(180deg,#ffffff,#f7fbff); padding:12px; border-radius:10px; box-shadow: 0 2px 6px rgba(15,76,129,0.07); }
    .small { font-size:12px; color:#555; }
    </style>
    """, unsafe_allow_html=True)

import subprocess, sys, time
from src import ai_modules
import io, base64


def ensure_models(ROOT, MODEL_DIR):
    """Ensure model artifacts exist. If not, run training script (src/train_model.py) to create them."""
    needed = ["power_theft_model.joblib", "scaler.joblib", "feature_columns.joblib"]
    missing = [f for f in needed if not Path(MODEL_DIR, f).exists()]

    if not missing:
        return True, "All model artifacts present."

    cmd = [sys.executable, "src/train_model.py", "--fast"]

    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
        out = proc.stdout + "\n" + proc.stderr

        ok = proc.returncode == 0 and all(Path(MODEL_DIR, f).exists() for f in needed)
        if ok:
            return True, out
        else:
            return False, "Training completed but artifacts missing.\n" + out

    except Exception as e:
        return False, f"Exception while running training: {e}"


try:
    ROOT = os.path.dirname(__file__)
    MODEL_DIR = os.path.join(ROOT, "models")
except Exception:
    ROOT = os.getcwd()
    MODEL_DIR = os.path.join(ROOT, "models")

ROOT = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT, "data", "meters_data.csv")

st.title("⚡ AI-Based Power Theft Detection — Dashboard (Prototype)")

# Load sample data
@st.cache_data(ttl=600)
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        st.error("Sample data not found. Run src/generate_data.py to create data/meters_data.csv")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=['Timestamp'])
    return df

df = load_data()

# Sidebar controls
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload CSV (Meter_ID,Timestamp,Consumption_kWh,Latitude,Longitude)", type=['csv'])
use_sample = st.sidebar.checkbox("Use sample data", value=True)
refresh = st.sidebar.checkbox("Auto-refresh every 5 seconds (demo)", value=False)
model_choice = st.sidebar.selectbox("Choose classifier for batch prediction", ["RandomForest", "LogisticRegression", "SVC"])
show_map = st.sidebar.checkbox("Show GIS Map (requires streamlit-folium & folium)", value=False)

if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=['Timestamp'])

if df.empty:
    st.warning("No data available")
    st.stop()

st.sidebar.markdown("### Quick stats")
st.sidebar.metric("Meters", df['Meter_ID'].nunique())
st.sidebar.metric("Total Rows", len(df))

# Retrain controls
st.sidebar.markdown("### Model Retrain (advanced)")
if st.sidebar.button("Retrain (fast demo)"):
    with st.spinner("Running fast retrain..."):
        ok, log = ensure_models(ROOT, MODEL_DIR)
        if ok:
            st.success("Fast retrain completed. Reloading model...")
            st.experimental_rerun()
        else:
            st.error("Retrain failed. See log below.")
            st.code(log[:10000])

if st.sidebar.button("Full retrain (slow)"):
    with st.spinner("Running full retrain (slow)..."):
        cmd = [sys.executable, "src/train_model.py", "--full"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
        out = proc.stdout + "\n" + proc.stderr

        if proc.returncode == 0:
            st.success("Full retrain finished.")
            st.experimental_rerun()
        else:
            st.error("Full retrain failed.")
            st.code(out[:10000])


# Feature engineering
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

        night_consumption = group[group['hour'].between(0, 5)]['Consumption_kWh'].mean()
        day_consumption = group[group['hour'].between(8, 20)]['Consumption_kWh'].mean()
        meter_data['night_day_ratio'] = (night_consumption / day_consumption) if day_consumption > 0 else 0

        hourly_avg = group.groupby('hour')['Consumption_kWh'].mean()
        meter_data['hourly_profile_std'] = hourly_avg.std()

        weekday_avg = group[group['day_of_week'] < 5]['Consumption_kWh'].mean()
        weekend_avg = group[group['day_of_week'] >= 5]['Consumption_kWh'].mean()
        meter_data['weekend_weekday_ratio'] = (weekend_avg / weekday_avg) if weekday_avg > 0 else 0

        meter_data['skewness'] = group['Consumption_kWh'].skew()
        meter_data['kurtosis'] = group['Consumption_kWh'].kurtosis()

        features.append(meter_data)

    return pd.DataFrame(features).set_index('Meter_ID').fillna(0)

features = create_features(df)

# Ensure models
success, train_log = ensure_models(ROOT, MODEL_DIR)
if not success:
    st.warning("Model artifacts were missing; attempted to train but failed.")
    st.code(train_log[:10000])
else:
    if "Saved model" in train_log or "Accuracy" in train_log:
        st.success("Model artifacts generated by auto-training.")

# Load model
model_path = os.path.join(MODEL_DIR, "power_theft_model.joblib")
scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
feat_cols_path = os.path.join(MODEL_DIR, "feature_columns.joblib")

model = scaler = feat_cols = None

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feat_cols = joblib.load(feat_cols_path)
        st.sidebar.success("Model loaded.")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
else:
    st.sidebar.info("No trained model found.")

# Batch prediction
results = None
if model is not None and scaler is not None:
    X = features.copy()
    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)
    y_prob = model.predict_proba(Xs)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(Xs))

    results = X.reset_index().assign(Predicted_Theft=y_pred, Theft_Prob=y_prob)
    st.write("### Batch Predictions (sample)")
    st.dataframe(results.head())

# Timeseries selection
st.sidebar.markdown("---")
meter_select = st.sidebar.selectbox("Select meter for timeseries", df['Meter_ID'].unique())
meter_df = df[df['Meter_ID'] == meter_select].sort_values('Timestamp').reset_index(drop=True)

st.header(f"Meter: {meter_select} — Time Series & Anomaly Timeline")

# Forecasting
use_prophet = False
try:
    from prophet import Prophet
    use_prophet = True
except:
    try:
        from prophet import Prophet
        use_prophet = True
    except:
        use_prophet = False

if use_prophet:
    ts = meter_df[['Timestamp', 'Consumption_kWh']].rename(columns={'Timestamp': 'ds', 'Consumption_kWh': 'y'})
    m = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=True)
    m.fit(ts)
    future = m.make_future_dataframe(periods=24 * 7, freq='H')
    forecast = m.predict(future)
    fig = m.plot(forecast)
    st.pyplot(fig)
else:
    meter_df['rolling_mean_24h'] = meter_df['Consumption_kWh'].rolling(window=24, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(meter_df['Timestamp'], meter_df['Consumption_kWh'], label='Observed', alpha=0.6)
    ax.plot(meter_df['Timestamp'], meter_df['rolling_mean_24h'], label='24h Rolling Mean', linestyle='--')
    ax.set_xlabel('Timestamp'); ax.set_ylabel('kWh'); ax.legend(); ax.grid(True)
    st.pyplot(fig)

# Anomalies
st.subheader("Detected Anomalies (basic rules)")
anoms = []

# zero streak > 48h
zeros = (meter_df['Consumption_kWh'] == 0)
zero_groups = (zeros != zeros.shift(1)).cumsum()
for _, g in meter_df.groupby(zero_groups):
    if g['Consumption_kWh'].eq(0).all() and len(g) >= 48:
        anoms.append({'type': 'zero_streak', 'start': g['Timestamp'].iloc[0], 'end': g['Timestamp'].iloc[-1]})

# night spikes
night_avg = meter_df[meter_df['Timestamp'].dt.hour.isin(range(0, 6))]['Consumption_kWh'].mean()
day_avg = meter_df[meter_df['Timestamp'].dt.hour.isin(range(8, 20))]['Consumption_kWh'].mean()
if day_avg > 0 and night_avg / day_avg > 3:
    anoms.append({'type': 'night_spike', 'description': f'Night avg {night_avg:.2f} kWh >> day {day_avg:.2f} kWh'})

# dips
deltas = meter_df['Consumption_kWh'].pct_change().fillna(0)
for i in meter_df.index[deltas < -0.9]:
    anoms.append({'type': 'sudden_dip', 'time': meter_df.loc[i, 'Timestamp'], 'value': meter_df.loc[i, 'Consumption_kWh']})

if anoms:
    for a in anoms:
        st.warning(a)
else:
    st.success("No simple-rule anomalies found.")

# GIS MAP
if show_map:
    try:
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=11)
        for mid, group in df.groupby('Meter_ID'):
            lat, lon = group['Latitude'].iloc[0], group['Longitude'].iloc[0]
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                popup=str(mid),
                color='red' if group['Consumption_kWh'].mean() < 0.2 else 'blue'
            ).add_to(m)

        st_folium(m, width=700)

    except Exception as e:
        st.error("Folium or streamlit-folium missing. Install with `pip install folium streamlit-folium`. " + str(e))

# Monthly bill estimator
st.subheader("Monthly Bill Estimator (last 30 days)")
last30 = meter_df[meter_df['Timestamp'] >= (meter_df['Timestamp'].max() - pd.Timedelta(days=30))]
total_kwh = last30['Consumption_kWh'].sum()

rate = st.sidebar.number_input("Rate per kWh (₹)", value=6.5, step=0.5)
est_bill = total_kwh * rate

st.metric("Estimated last 30d kWh", f"{total_kwh:.2f} kWh")
st.metric("Estimated bill (last 30d)", f"₹{est_bill:.2f}")

# SHAP
st.subheader("Model Explanation (SHAP)")
try:
    import shap
    if model is not None:
        explainer = shap.Explainer(model.predict_proba, scaler.transform(features) if scaler is not None else features)
        shap_values = explainer(features)
        st.write("SHAP available — summary for first 3 meters")
        st.pyplot(shap.plots.beeswarm(shap_values[:3]))
    else:
        st.info("Model not loaded.")
except:
    st.info("SHAP not installed.")

# Severity scoring
st.subheader("Theft Severity Score (derived)")
if results is not None:
    def severity_label(row):
        prob = row['Theft_Prob']
        score = 0
        score += min(row.get('zero_consumption_hours', 0) / 24.0, 1.0) * 2.0
        score += min(row.get('night_day_ratio', 0), 5.0) * 1.5
        score += (1.0 - min(row.get('hourly_profile_std', 1.0), 1.0)) * 1.0
        score += prob * 2.0

        if score < 2.5: return 'Low'
        if score < 5.0: return 'Medium'
        if score < 7.5: return 'High'
        return 'Critical'

    results['Severity'] = results.apply(severity_label, axis=1)
    st.dataframe(results[['Meter_ID', 'Predicted_Theft', 'Theft_Prob', 'Severity']]
                 .sort_values('Theft_Prob', ascending=False)
                 .head(20))
else:
    st.info("Run training to see severity.")

# ---- AI Suite ----
st.header("AI Suite — Advanced features")

tab1, tab2, tab3, tab4 = st.tabs(["LLM Explanation", "Forecasting", "Clustering & Ensemble", "Realtime Alerts"])

# === TAB 1 ===
with tab1:
    st.subheader("LLM-style Explanation (templated)")
    meter_sel = st.selectbox("Choose meter ID", options=sorted(df['Meter_ID'].unique()))
    meter_df = df[df['Meter_ID'] == meter_sel]

    feat = {}
    feat['night_day_ratio'] = (
        meter_df[meter_df['Timestamp'].dt.hour.between(0, 5)]['Consumption_kWh'].mean() /
        (meter_df[meter_df['Timestamp'].dt.hour.between(8, 20)]['Consumption_kWh'].mean() + 1e-6)
    )
    feat['zero_consumption_hours'] = (meter_df['Consumption_kWh'] == 0).sum()
    feat['hourly_profile_std'] = meter_df.groupby(meter_df['Timestamp'].dt.hour)['Consumption_kWh'].mean().std()

    try:
        Xs = pd.DataFrame([feat])
        prob = 0.0
        if model is not None:
            prob_arr, vote_arr = ai_modules.predict_ensemble({'rf': model, 'scaler': scaler}, Xs)
            prob = float(prob_arr[0])
        else:
            prob = 0.5 if feat['zero_consumption_hours'] > 10 else 0.05

        explanation = ai_modules.llm_explain(meter_sel, feat, {'theft_prob': prob})

    except Exception as e:
        explanation = f"Error generating explanation: {e}"

    st.code(explanation)

# === TAB 2 ===
with tab2:
    st.subheader("Forecasting (Prophet or rolling mean)")
    meter_f = st.selectbox("Select meter for forecast", options=sorted(df['Meter_ID'].unique()))
    meter_df = df[df['Meter_ID'] == meter_f][['Timestamp', 'Consumption_kWh']]
    forecast = ai_modules.forecast_profile(meter_df, periods=24)
    st.line_chart(pd.DataFrame({'ds': forecast['ds'].astype(str), 'yhat': forecast['yhat']}).set_index('ds'))

# === TAB 3 ===
with tab3:
    st.subheader("Clustering & Ensemble")

    if st.button("Run clustering and train fast ensemble"):
        meters = []
        feats = []
        for mid, group in df.groupby('Meter_ID'):
            meters.append(mid)
            feats.append({
                'avg_consumption': group['Consumption_kWh'].mean(),
                'std_consumption': group['Consumption_kWh'].std(),
                'zero_hours': (group['Consumption_kWh'] == 0).sum()
            })
        feat_df = pd.DataFrame(feats, index=meters)

        labels, km = ai_modules.cluster_meters(feat_df, n_clusters=4)
        st.write("Cluster assignments (sample):", dict(zip(meters[:10], labels[:10])))

        if 'Theft_Flag' in df.columns:
            X = feat_df.fillna(0)
            y = [int(df[df['Meter_ID'] == mid]['Theft_Flag'].mode().iloc[0]) for mid in feat_df.index]

            models_local = ai_modules.train_ensemble(X, np.array(y), use_xgb=False)
            st.session_state['ensemble_models'] = models_local
            st.success("Trained fast ensemble.")
        else:
            st.info("No Theft_Flag found for supervised training.")

    if 'ensemble_models' in st.session_state:
        st.write("Ensemble ready.")
        if st.button("Predict sample meters"):
            sample_feats = feat_df.fillna(0).iloc[:5]
            probs, votes = ai_modules.predict_ensemble(st.session_state['ensemble_models'], sample_feats)
            st.write("Probs:", probs.tolist(), "Votes:", votes.tolist())

# === TAB 4 ===
with tab4:
    st.subheader("Realtime Alerts Simulation")

    if st.button("Start simulation (50 steps)"):
        gen = ai_modules.simulate_realtime_alerts(df, steps=50)
        alerts = []

        for item in gen:
            mid = item['Meter_ID']
            val = item['Consumption_kWh']
            mean_val = df[df['Meter_ID'] == mid]['Consumption_kWh'].mean()

            if val < 0.2 * mean_val:
                alerts.append((mid, item['Timestamp'], val))

        if alerts:
            st.warning(f"{len(alerts)} alerts simulated. Sample: {alerts[:5]}")
        else:
            st.success("No alerts in simulation.")

# Auto-refresh
if refresh:
    st.info("Auto-refreshing dashboard every 5 seconds (demo).")
    time.sleep(5)
    st.experimental_rerun()
