"""
ai_modules.py - AI feature modules for Vidyutrakshak
Provides:
 - llm_explain: templated explanation of suspicious meter behavior (can be replaced with real LLM call)
 - forecast_profile: uses Prophet if available, otherwise rolling mean
 - train_ensemble: trains RandomForest (+ XGBoost if installed) and returns fitted models
 - predict_ensemble: returns prob and vote
 - cluster_meters: KMeans clustering on feature matrix
 - generate_report: creates a simple HTML report (downloadable) with visuals and LLM explanation
 - simulate_realtime_alerts: generator that yields simulated readings (for demo)
"""

import os, io, json, datetime, base64
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

def llm_explain(meter_id, features, probs):
    """
    Return a human-readable explanation for a suspected theft.
    This is a templated fallback; replace with an LLM call for richer text.
    """
    p = probs.get('theft_prob', None)
    lines = []
    lines.append(f"Meter {meter_id} analysis:")
    if p is None:
        lines.append("No theft probability provided.")
    else:
        lines.append(f"- Theft probability: {p:.2f}")
    # examine features heuristically
    if 'night_day_ratio' in features and features['night_day_ratio'] < 0.2:
        lines.append("- Low night/day consumption ratio: possible bypass during nights.")
    if 'zero_consumption_hours' in features and features['zero_consumption_hours'] > 10:
        lines.append(f"- High zero-consumption hours ({features['zero_consumption_hours']} hrs): meter likely tampered or disconnected intermittently.")
    if 'hourly_profile_std' in features and features['hourly_profile_std'] < 0.05:
        lines.append("- Very flat hourly profile: suspicious smoothing or meter manipulation.")
    if len(lines) == 1:
        lines.append("- Pattern unclear; further investigation recommended.")
    return "\\n".join(lines)

def forecast_profile(df_meter, periods=24):
    """
    Forecast next `periods` hours for a single meter's series.
    If prophet installed, use it; otherwise fallback to rolling mean.
    df_meter: DataFrame with columns ['Timestamp','Consumption_kWh']
    """
    df = df_meter.copy().rename(columns={'Timestamp':'ds','Consumption_kWh':'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    if _HAS_PROPHET and len(df) > 10:
        m = Prophet()
        m.fit(df[['ds','y']])
        future = m.make_future_dataframe(periods=periods, freq='H')
        fcst = m.predict(future)
        return fcst[['ds','yhat']].tail(periods)
    else:
        # rolling mean
        df = df.set_index('ds').resample('H').mean().ffill()
        mean = df['y'].rolling(24, min_periods=1).mean().iloc[-1]
        idx = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=periods, freq='H')
        return pd.DataFrame({'ds': idx, 'yhat': [mean]*periods})

def train_ensemble(X, y, use_xgb=False):
    """
    Train RandomForest and optionally XGBoost. Returns dict with models and scaler.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(Xs, y)
    models = {'rf': rf, 'scaler': scaler}
    if use_xgb and _HAS_XGBOOST:
        xg = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xg.fit(X, y)
        models['xgb'] = xg
    return models

def predict_ensemble(models, X):
    """
    Return average probability and majority vote across models present.
    X should be raw (unscaled) features DataFrame.
    """
    scaler = models.get('scaler')
    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        Xs = X.values
    probs = {}
    votes = []
    # RF
    if 'rf' in models:
        p = models['rf'].predict_proba(Xs)[:,1]
        probs['rf'] = p
        votes.append(models['rf'].predict(Xs))
    # XGB
    if 'xgb' in models and _HAS_XGBOOST:
        p2 = models['xgb'].predict_proba(X.values)[:,1]
        probs['xgb'] = p2
        votes.append(models['xgb'].predict(X.values))
    # aggregate
    prob_mean = np.mean(np.vstack(list(probs.values())), axis=0) if len(probs)>0 else np.zeros(X.shape[0])
    # majority vote
    if votes:
        votes_arr = np.vstack(votes)
        maj_vote = np.apply_along_axis(lambda r: np.bincount(r).argmax(), axis=0, arr=votes_arr)
    else:
        maj_vote = np.zeros(X.shape[0], dtype=int)
    return prob_mean, maj_vote

def cluster_meters(feature_df, n_clusters=5):
    """Cluster meters using KMeans on provided features (rows=meter)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(feature_df.fillna(0))
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(Xs)
    return labels, km

def generate_report(meter_id, meter_df, features, preds, explanation_text, outpath=None, pdf=False):
    """
    Generate a simple HTML report with summary, small plots (inline base64), and explanation text.
    Returns path to saved HTML file or HTML string if outpath None.
    """
    import matplotlib.pyplot as plt
    # plot timeseries
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(pd.to_datetime(meter_df['Timestamp']), meter_df['Consumption_kWh'])
    ax.set_title(f"Meter {meter_id} Consumption")
    ax.set_xlabel("Time"); ax.set_ylabel("kWh")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    html = f\"\"\"
    <html><head><meta charset='utf-8'><title>Report - Meter {meter_id}</title></head><body>
    <h1>Forensic Report - Meter {meter_id}</h1>
    <h2>Summary</h2>
    <p>Predicted theft probability: {preds:.2f}</p>
    <h2>Explanation</h2>
    <pre>{explanation_text}</pre>
    <h2>Consumption Plot</h2>
    <img src='data:image/png;base64,{data}' width='800'/>
    <h2>Features</h2>
    <pre>{json.dumps(features, indent=2)}</pre>
    </body></html>
    \"\"\"
    if outpath:
        with open(outpath, 'w', encoding='utf-8') as f:
            f.write(html)
        # optionally create PDF using reportlab if requested
        if pdf:
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfgen import canvas
                from reportlab.lib.utils import ImageReader
                import tempfile
                # create simple PDF with title and embedded PNG
                tmpimg = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                tmpimg.write(base64.b64decode(data))
                tmpimg.flush()
                pdf_path = os.path.splitext(outpath)[0] + '.pdf'
                c = canvas.Canvas(pdf_path, pagesize=letter)
                c.setFont('Helvetica-Bold', 16)
                c.drawString(72, 750, f'Forensic Report - Meter {meter_id}')
                # draw image
                img = ImageReader(tmpimg.name)
                c.drawImage(img, 72, 350, width=450, height=200)
                c.setFont('Helvetica', 10)
                text = explanation_text[:2000]
                c.drawString(72, 320, 'Summary:')
                c.drawString(72, 300, text)
                c.save()
                return outpath, pdf_path
            except Exception as e:
                # fallback - return HTML path and error
                return outpath, str(e)
        return outpath
    return html

def simulate_realtime_alerts(df, meters=None, steps=50):
    """
    Simple generator that yields (meter_id, Timestamp, Consumption_kWh) tuples to simulate streaming data.
    If meters None, picks random meters from df.
    """
    if meters is None:
        meters = df['Meter_ID'].unique().tolist()
    rng = np.random.default_rng(42)
    for _ in range(steps):
        mid = rng.choice(meters)
        ts = datetime.datetime.now().isoformat()
        # pick a recent consumption value and add noise; lower it sometimes to simulate theft
        subset = df[df['Meter_ID']==mid]
        if subset.empty:
            continue
        val = float(subset['Consumption_kWh'].sample(1).iloc[0])
        if rng.random() < 0.05:
            val = val * rng.uniform(0.0, 0.2)  # simulate dip/theft
        else:
            val = val * rng.uniform(0.9, 1.2)
        yield {'Meter_ID': int(mid), 'Timestamp': ts, 'Consumption_kWh': float(val)}


# Optional OpenAI integration: if OPENAI_API_KEY env var is set and 'openai' package is installed,
# ai_modules.llm_call will send prompt and return the model response. Falls back to templated llm_explain.
def llm_call(prompt, model="gpt-4o-mini", max_tokens=512, temperature=0.2):
    try:
        import os
        if os.environ.get("OPENAI_API_KEY") is None:
            return None, "No OPENAI_API_KEY set; skipping remote LLM call."
        import openai
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":"You are an assistant that explains electricity meter anomalies concisely."},
                      {"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        text = resp['choices'][0]['message']['content'].strip()
        return text, None
    except Exception as e:
        return None, str(e)

# Enhanced llm_explain: try remote LLM first, else fallback to template explanation
def llm_explain_with_model(meter_id, features, probs, use_remote=True):
    tmpl = llm_explain(meter_id, features, probs)
    if use_remote:
        prompt = f\"\"\"Analyze the following meter features and provide a short forensic-style explanation and recommended next actions.\nMeter: {meter_id}\nFeatures: {json.dumps(features)}\nPredictions: {json.dumps(probs)}\nKeep it under 150 words.\"\"\"
        text, err = llm_call(prompt)
        if text:
            return text
        else:
            return tmpl + "\\n\\n(LLM call failed: " + str(err) + ")"
    else:
        return tmpl
