"""
generate_data.py
Generates synthetic hourly meter data with normal and suspicious (theft) patterns.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
import os

def generate(outfile=None, n_meters=200, days=90, seed=42):
    # default outfile path: repository root / data / meters_data.csv
    if outfile is None:
        ROOT = os.path.dirname(os.path.dirname(__file__))
        outfile = os.path.join(ROOT, "data", "meters_data.csv")
    np.random.seed(seed)
    start = datetime.now() - timedelta(days=days)
    periods = 24 * days
    timestamps = [start + timedelta(hours=i) for i in range(periods)]
    rows = []
    for m in range(1, n_meters + 1):
        lat = 28.50 + np.random.rand() * 0.5
        lon = 77.0 + np.random.rand() * 0.6
        base = 0.5 + np.random.rand() * 1.5
        daily_variation = 0.5 + np.random.rand() * 2.5
        seasonal = 1 + 0.2 * np.sin(np.linspace(0, 6*np.pi, periods) + np.random.rand()*5)
        noise = np.random.normal(0, 0.05, periods)
        hours = np.array([ts.hour for ts in timestamps])
        profile = base + daily_variation * (0.3 + 0.7 * (np.exp(-((hours-19)%24-5)**2/15) + np.exp(-((hours-8)%24-2)**2/8)))
        consumption = profile * seasonal + noise
        consumption[consumption < 0] = 0.01
        for i, ts in enumerate(timestamps):
            rows.append([f"M{m:04d}", ts, float(consumption[i]), lat, lon])
    df = pd.DataFrame(rows, columns=['Meter_ID', 'Timestamp', 'Consumption_kWh', 'Latitude', 'Longitude'])
    # simple theft injection (small percent)
    meter_ids = df['Meter_ID'].unique()
    theft_ids = np.random.choice(meter_ids, size=int(0.12 * len(meter_ids)), replace=False)
    df['Theft_Flag'] = 0
    for tid in theft_ids:
        mask = df['Meter_ID'] == tid
        idxs = df.loc[mask].index
        start_idx = np.random.randint(0, len(idxs)-24*5)
        mode = np.random.choice(['zero','flat_low','night_spike','dips'])
        if mode == 'zero':
            df.loc[idxs[start_idx:start_idx+24*3], 'Consumption_kWh'] = 0.0
        elif mode == 'flat_low':
            df.loc[idxs[start_idx:start_idx+24*15], 'Consumption_kWh'] *= np.random.uniform(0.05, 0.25)
        elif mode == 'night_spike':
            seg = idxs[start_idx:start_idx+24*10]
            for j in seg:
                if df.loc[j, 'Timestamp'].hour in range(0,6):
                    df.loc[j, 'Consumption_kWh'] *= (1 + np.random.rand()*3)
        else:
            dips = np.random.choice(idxs, size=20, replace=False)
            df.loc[dips, 'Consumption_kWh'] *= np.random.uniform(0.0, 0.05, size=len(dips))
        df.loc[df['Meter_ID'] == tid, 'Theft_Flag'] = 1
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_csv(outfile, index=False)
    print(f"Saved synthetic data to {outfile} (meters={len(meter_ids)}, rows={len(df)})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/meters_data.csv")
    parser.add_argument("--meters", type=int, default=200)
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    generate(outfile=args.out, n_meters=args.meters, days=args.days)