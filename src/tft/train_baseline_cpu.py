"""Quick baseline regression on CPU to match XGBoost-like outputs (MAE/RMSE/R2)
This uses scikit-learn LinearRegression on lag features to predict the next timestep generation.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "multi_home_15min.csv"

if not DATA_FILE.exists():
    raise SystemExit(f"Sample data not found: {DATA_FILE}")

print(f"Loading {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

# map pv/load columns
if 'generation' not in df.columns:
    if 'pv_kw' in df.columns:
        df['generation'] = df['pv_kw']
    else:
        df['generation'] = 0.0

if 'house_id' not in df.columns:
    df['house_id'] = 1

# sort
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['house_id','timestamp']).reset_index(drop=True)

# Create lag features (prev 4 timesteps) and next-step target
LAGS = 4
rows = []
for house, g in df.groupby('house_id'):
    vals = g['generation'].values
    for i in range(LAGS, len(vals)-1):
        row = {}
        for lag in range(1, LAGS+1):
            row[f'lag_{lag}'] = vals[i-lag]
        row['target'] = vals[i+1]
        row['house_id'] = house
        row['timestamp'] = g['timestamp'].iloc[i]
        rows.append(row)

if not rows:
    raise SystemExit('Not enough data to build lag features')

Xy = pd.DataFrame(rows)

# train/val/test split by time
Xy = Xy.sort_values('timestamp')
features = [f'lag_{i}' for i in range(1,LAGS+1)]
Xy = Xy.dropna(subset=features+['target'])
cut1 = int(0.7*len(Xy))
cut2 = int(0.85*len(Xy))
train = Xy.iloc[:cut1]
val = Xy.iloc[cut1:cut2]
test = Xy.iloc[cut2:]

features = [f'lag_{i}' for i in range(1,LAGS+1)]

model = LinearRegression()
model.fit(train[features], train['target'])

for name, ds in [('train',train), ('val',val), ('test',test)]:
    preds = model.predict(ds[features])
    mae = mean_absolute_error(ds['target'], preds)
    rmse = np.sqrt(mean_squared_error(ds['target'], preds))
    r2 = r2_score(ds['target'], preds)
    print(f"{name} | rows={len(ds)} | MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

# show first 10 predictions vs truth
preds = model.predict(test[features])
print('\nFirst 10 predictions vs truth:')
print(pd.DataFrame({'pred':preds[:10], 'true':test['target'].values[:10]}))
