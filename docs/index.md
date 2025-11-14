# Power Theft Detection

# AI-Based Power Theft Pattern Detection Using Load Signature Analysis

This repository contains a prototype end-to-end project for detecting electricity theft using load signature analysis and ML.
Features included in this deliverable (prototype level):
- Synthetic dataset generator (hourly readings, lat/lon, theft injection)
- Feature engineering pipeline (many signature features)
- RandomForest baseline training script and artifact saving
- Streamlit dashboard with:
  - Timeseries + forecasting (Prophet optional fallback)
  - Anomaly timeline visualization (zero streaks, night spikes, dips)
  - GIS map of meters (folium optional)
  - Severity scoring, monthly bill estimator
  - SHAP explanations (optional)
  - Multi-model comparison hooks
- Files: README, requirements.txt, .gitignore, sample data, source scripts, app

## How to run (Windows)

1. Create virtual environment and activate
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install requirements
```powershell
pip install -r requirements.txt
```

3. Generate data (optional, sample data already included)
```powershell
python src/generate_data.py --out data/meters_data.csv --meters 200 --days 90
```

4. Train baseline model (creates models/power_theft_model.joblib)
```powershell
python src/train_model.py
```

5. Run Streamlit dashboard
```powershell
streamlit run app.py
```

## Notes & Next Steps
- Prophet and SHAP are optional dependencies; if not installed the app will fall back to simpler baselines.
- For LSTM forecasting integration, see `src/train_model.py` comments — you can add a Keras LSTM model to forecast expected load per-meter and compare real vs predicted.
- For production deployments, store large models/data using Git LFS or cloud storage (S3).

## License
MIT
