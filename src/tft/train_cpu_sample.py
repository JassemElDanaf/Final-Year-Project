# Small CPU-only sample trainer for TFT
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data import GroupNormalizer

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "multi_home_15min.csv"

if not DATA_FILE.exists():
    raise SystemExit(f"Sample data not found: {DATA_FILE}")

print(f"Loading sample data from {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
print('Raw columns:', list(df.columns))

# Map common column names to the training expectations
if 'generation' not in df.columns:
    if 'pv_kw' in df.columns:
        df['generation'] = df['pv_kw']
    elif 'pv' in df.columns:
        df['generation'] = df['pv']
    else:
        df['generation'] = 0.0

if 'target' not in df.columns:
    df['target'] = df['generation'].astype('float32')

# usage mapping
if 'usage' not in df.columns:
    if 'load_kw' in df.columns:
        df['usage'] = df['load_kw']
    elif 'load' in df.columns:
        df['usage'] = df['load']

# ensure house_id exists
if 'house_id' not in df.columns:
    df['house_id'] = 1

# timestamp & time_idx
if 'timestamp' not in df.columns and 'datetime' in df.columns:
    df['timestamp'] = df['datetime']

df['timestamp'] = pd.to_datetime(df['timestamp'])
# create time_idx based on minutes cadence (infer from data)
# derive cadence from unique timestamps (handle duplicates across houses)
uniques = df['timestamp'].sort_values().drop_duplicates()
if len(uniques) < 2:
    raise SystemExit('Not enough unique timestamps in data')
median_diff = uniques.diff().median()
step_minutes = int(median_diff.total_seconds() // 60)
if step_minutes <= 0:
    print('Could not infer cadence, defaulting to 15 minutes')
    step_minutes = 15
print('Inferred step minutes:', step_minutes)

# create integer time_idx (based on epoch / step)
df['time_idx'] = (df['timestamp'].astype('int64') // (step_minutes*60*10**9)).astype('int64')

# time features
df['hour'] = df['timestamp'].dt.hour
df['dow'] = df['timestamp'].dt.weekday
df['is_weekend'] = (df['dow'] >= 5).astype('int8')
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24).astype('float32')
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24).astype('float32')

# drop rows with NaN target
df = df.dropna(subset=['target']).copy()

print('Data shape after processing:', df.shape)

# make house_id categorical string as required by TimeSeriesDataSet
df['house_id'] = df['house_id'].astype(str)

# small train/val/test split by time_idx
tmin, tmax = int(df['time_idx'].min()), int(df['time_idx'].max())
train_cut = int(tmin + 0.7*(tmax - tmin))
val_cut = int(tmin + 0.85*(tmax - tmin))
train_df = df[df.time_idx <= train_cut].copy()
val_df = df[(df.time_idx > train_cut) & (df.time_idx <= val_cut)].copy()
test_df = df[df.time_idx > val_cut].copy()

print('Train/val/test sizes:', train_df.shape, val_df.shape, test_df.shape)

# dataset windows
ENC_DAYS = 1
PRED_DAYS = 1
max_encoder_length = int(24*60//step_minutes * ENC_DAYS)
max_prediction_length = int(24*60//step_minutes * PRED_DAYS)
print('Encoder/prediction lengths:', max_encoder_length, max_prediction_length)

training = TimeSeriesDataSet(
    train_df,
    time_idx='time_idx',
    target='target',
    group_ids=['house_id'],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    # for a quick CPU test skip target normalizer to avoid version-specific transform issues
    target_normalizer=None,
    allow_missing_timesteps=True,
    static_categoricals=['house_id'],
    time_varying_known_reals=['hour_sin','hour_cos','dow','is_weekend'],
    time_varying_unknown_reals=['target','generation'],
)

validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)

train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

# small model for CPU
tft = TemporalFusionTransformer.from_dataset(
    training,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    learning_rate=1e-3,
    optimizer='adam',
    loss=MAE()
)

seed_everything(42)

print('Model type:', type(tft))
try:
    import pytorch_lightning as pl
    print('Is LightningModule:', isinstance(tft, pl.LightningModule))
except Exception:
    pass

trainer = Trainer(
    max_epochs=2,
    accelerator='cpu',
    devices=1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
    log_every_n_steps=20,
)

print('Starting training (CPU) ...')
trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
y_true = []
y_pred = []
# show callback metrics from trainer (contains val_loss etc.)
print('Trainer callback metrics:', trainer.callback_metrics)

# run a quick prediction on validation and print a small sample
print('Running quick validation predictions...')
preds = tft.predict(val_loader)
print('Predictions shape:', np.array(preds).shape)
print('First 5 predictions (first horizon):', np.array(preds)[:5, 0])

print('Done.')
