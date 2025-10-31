"""Train a simple hourly multi-output LinearRegression baseline to predict next-day (24h) PV per house.
Compares results to existing xgboost hourly predictions in data/processed/xgboost/agent1_predictions.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "multi_home_15min.csv"
XGB_FILE = ROOT / "data" / "processed" / "xgboost" / "agent1_predictions.csv"

HORIZON = 24  # next-day hourly
LAGS = 24     # use past 24 hours as features

if not DATA_FILE.exists():
    raise SystemExit(f"Sample data not found: {DATA_FILE}")
if not XGB_FILE.exists():
    print(f"Warning: xgboost predictions not found at {XGB_FILE}. Comparison will be skipped.")

print('Loading data...')
df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
# resample to hourly per house by mean
print('Resampling to hourly...')
# use groupby-resample without duplicating the house_id column
df_hourly = df.set_index('timestamp')
df_hourly = df_hourly.groupby('house_id', group_keys=False).resample('1H').mean().reset_index()
# focus on pv generation column
if 'generation' not in df_hourly.columns:
    if 'pv_kw' in df_hourly.columns:
        df_hourly['pv'] = df_hourly['pv_kw']
    elif 'pv_nextday_kw_pred' in df_hourly.columns:
        df_hourly['pv'] = df_hourly['pv_nextday_kw_pred']
    else:
        # try pv_kw earlier
        df_hourly['pv'] = df_hourly.get('pv_kw', 0.0)
else:
    df_hourly['pv'] = df_hourly['generation']

# build samples per house
rows = []
for house, g in df_hourly.groupby('house_id'):
    g = g.sort_values('timestamp').reset_index(drop=True)
    vals = g['pv'].values
    times = g['timestamp'].values
    for i in range(LAGS, len(vals)-HORIZON):
        feat = vals[i-LAGS:i]
        target = vals[i:i+HORIZON]
        rows.append({'house_id':house, 'timestamp': times[i], 'feat': feat, 'target': target})

if not rows:
    raise SystemExit('Not enough data to build hourly multi-horizon samples')

print('Building arrays...')
X = np.stack([r['feat'] for r in rows])
y = np.stack([r['target'] for r in rows])
houses = np.array([r['house_id'] for r in rows])
timestamps = np.array([r['timestamp'] for r in rows])

print('Shapes X,y:', X.shape, y.shape)
# Drop any samples with NaN in features or targets
mask_finite = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
X = X[mask_finite]
y = y[mask_finite]
houses = houses[mask_finite]
timestamps = timestamps[mask_finite]
print('After dropping NaNs, shapes X,y:', X.shape, y.shape)

# train/val/test split by time
n = len(X)
cut1 = int(0.7*n)
cut2 = int(0.85*n)
X_train, X_val, X_test = X[:cut1], X[cut1:cut2], X[cut2:]
y_train, y_val, y_test = y[:cut1], y[cut1:cut2], y[cut2:]

print('Training multi-output LinearRegression...')
model = LinearRegression()
model.fit(X_train, y_train)

print('Predicting...')
preds_val = model.predict(X_val)
preds_test = model.predict(X_test)

# compute per-horizon MAE
mae_per_horizon = np.mean(np.abs(preds_test - y_test), axis=0)
rmse_per_horizon = np.sqrt(np.mean((preds_test - y_test)**2, axis=0))

print('Overall test MAE:', mae_per_horizon.mean())
print('Overall test RMSE:', rmse_per_horizon.mean())

print('\nPer-horizon MAE (first 24):')
for h in range(HORIZON):
    print(f'h={h+1:02d} | MAE={mae_per_horizon[h]:.4f} | RMSE={rmse_per_horizon[h]:.4f}')

# Compare with xgboost if available
if XGB_FILE.exists():
    print('\nComparing to xgboost predictions...')
    xgb = pd.read_csv(XGB_FILE, parse_dates=['timestamp'])
    # we need to align by house and timestamp where xgb has pv_nextday_kw_pred for each future hour
    # Build our predictions into a long table for the same timestamps
    # For each sample in test set, timestamp represents the time of prediction; xgb likely lists predicted times for each future hour.
    # We'll compare aggregated daily MAE by aligning timestamps + horizons
    # Build long df for our preds_test
    long_rows = []
    for i in range(len(preds_test)):
        ts = timestamps[cut2 + i]
        house = houses[cut2 + i]
        for h in range(HORIZON):
            pred_time = pd.Timestamp(ts) + pd.Timedelta(hours=h+1)
            long_rows.append({'timestamp': pred_time, 'house_id': int(house), 'horizon': h+1, 'pred': preds_test[i,h], 'true': y_test[i,h]})
    long_df = pd.DataFrame(long_rows)
    # join with xgb predictions per timestamp+house
    xgb_sub = xgb[['timestamp','house_id','pv_nextday_kw_pred']].rename(columns={'pv_nextday_kw_pred':'xgb_pred'})
    merged = pd.merge(long_df, xgb_sub, on=['timestamp','house_id'], how='left')
    # drop rows without xgb
    merged = merged.dropna(subset=['xgb_pred'])
    # compute MAE per method
    mae_ours = np.mean(np.abs(merged['pred'] - merged['true']))
    mae_xgb = np.mean(np.abs(merged['xgb_pred'] - merged['true']))
    print(f'Comparison on overlapping timestamps: our MAE={mae_ours:.4f} | xgb MAE={mae_xgb:.4f}')
    # per-horizon compare
    for h in range(1,HORIZON+1):
        m = merged[merged['horizon']==h]
        if len(m)==0: continue
        print(f'h={h:02d} | our MAE={np.mean(np.abs(m["pred"]-m["true"])):.4f} | xgb MAE={np.mean(np.abs(m["xgb_pred"]-m["true"])):.4f}')

print('\nDone.')
