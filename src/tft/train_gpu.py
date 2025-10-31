# -*- coding: utf-8 -*-
"""
Train a Temporal Fusion Transformer (TFT) on multi-home PV/load data with GPU acceleration.
"""

from pathlib import Path
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
# Try to import Trainer and callbacks from the newer `lightning` package, fall back to `pytorch_lightning`
try:
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
except Exception:
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import os


# ----------------- GPU CHECK -----------------
if not torch.cuda.is_available():
    raise RuntimeError("This script requires GPU support! Please install CUDA and PyTorch with CUDA support.")

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")


# ----------------- CONFIG -----------------
ROOT = Path(__file__).resolve().parents[2]      # repo root
RAW_IN = ROOT / "data" / "raw"                # raw data per house
DATA_IN = ROOT / "data" / "processed"         # (optional) processed tables
ART_DIR = ROOT / "models" / "tft"             # checkpoints
ART_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters - OPTIMIZED FOR MAXIMUM ACCURACY
# Strategy: Use ALL 3 years of data for training samples, but optimal encoder window
BATCH = int(os.environ.get("TFT_BATCH", 64))            # per-GPU batch size
PRECISION = os.environ.get("TFT_PRECISION", "32-true")    # use FP32 for stability (can override to 16-mixed if no issues)
EPOCHS = int(os.environ.get("TFT_EPOCHS", 100))         # increased for better convergence
PATIENCE = int(os.environ.get("TFT_PATIENCE", 15))      # early stopping patience
PRED_DAYS = int(os.environ.get("TFT_PRED_DAYS", 1))     # prediction length in days
CADENCE = os.environ.get("TFT_CADENCE", "1H")        # data timestep (recommended: hourly)
NUM_WORKERS = int(os.environ.get("TFT_NUM_WORKERS", 4))
ACCELERATOR = "gpu"
# use all available GPU devices by default
DEVICES = list(range(torch.cuda.device_count()))

# Data configuration - use all 3 years for training, optimal encoder length for accuracy
YEARS_OF_HISTORY = int(os.environ.get("TFT_HISTORY_YEARS", 3))  # all data for training samples
ENCODER_DAYS = int(os.environ.get("TFT_ENCODER_DAYS", 14))      # optimal: 14 days lookback

# Model architecture - optimized for accuracy
HIDDEN_SIZE = int(os.environ.get("TFT_HIDDEN", 160))            # increased capacity
ATTENTION_HEADS = int(os.environ.get("TFT_HEADS", 4))           # multi-head attention
DROPOUT = float(os.environ.get("TFT_DROPOUT", 0.15))            # regularization
LEARNING_RATE = float(os.environ.get("TFT_LR", 0.001))          # initial LR

# Dry-run flag (set env DRY_RUN=1 to avoid doing a long full training; default DRY_RUN=1 here to validate)
DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"

# Set random seed for reproducibility
seed_everything(42)
# tweak cudnn settings: disable cudnn to avoid some LSTM/cuDNN compatibility issues on certain GPUs
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


def _infer_timestamp_column(df):
    for c in ["timestamp", "Timestamp", "DateTime", "Date", "Date_Time"]:
        if c in df.columns:
            return c
    # fallback: first column
    return df.columns[0]


def _infer_pv_column(df):
    candidates = [
        "PV Power Generation (W)", "PV Power (W)", "PV", "pv_kW", "pv_kw", "generation", "Power (W)", "PV (W)"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: numeric column other than timestamp
    for c in df.columns:
        if c.lower().find("pv") >= 0 or c.lower().find("generation") >= 0:
            return c
    return None


def _infer_load_column(df):
    candidates = ["Consumption (kW)", "Load (kW)", "load_kw", "load_kW", "consumption"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: any numeric column not PV
    for c in df.columns:
        if df[c].dtype.kind in "fi" and c.lower().find("pv") < 0:
            return c
    return None


def read_house_folder(house_path: Path, house_id: int, cadence: str = CADENCE):
    """Read available raw CSVs in a house folder and return a tidy DataFrame with columns [timestamp, pv_kW, load_kW, house_id, temperature, irradiance]."""
    dfs = []
    # PV
    pv_files = list(house_path.glob("*PV*.csv"))
    load_files = list(house_path.glob("*Load*.csv"))
    weather_files = list(house_path.glob("*Weather*.csv"))
    
    # prefer explicit files if present
    if pv_files:
        pv = pd.read_csv(pv_files[0])
        tcol = _infer_timestamp_column(pv)
        pv[tcol] = pd.to_datetime(pv[tcol])
        pv_col = _infer_pv_column(pv)
        if pv_col is not None:
            # pv might be in W -> convert to kW if values look large
            pv_vals = pv[pv_col].astype(float)
            if pv_vals.max() > 1000:  # likely W
                pv_kW = pv_vals / 1000.0
            else:
                pv_kW = pv_vals
            pv_df = pv[[tcol]].copy()
            pv_df = pv_df.rename(columns={tcol: 'timestamp'})
            pv_df['pv_kW'] = pv_kW.values
            
            # Extract irradiance if available (GHI column)
            if 'GHI (W/m2)' in pv.columns:
                pv_df['irradiance'] = pv['GHI (W/m2)'].astype(float)
            
            dfs.append(pv_df)

    if load_files:
        ld = pd.read_csv(load_files[0])
        tcol = _infer_timestamp_column(ld)
        ld[tcol] = pd.to_datetime(ld[tcol])
        load_col = _infer_load_column(ld)
        if load_col is not None:
            load_df = ld[[tcol, load_col]].copy()
            load_df = load_df.rename(columns={tcol: 'timestamp', load_col: 'load_kW'})
            dfs.append(load_df)
    
    # Extract weather data (temperature, etc.)
    if weather_files:
        weather = pd.read_csv(weather_files[0])
        tcol = _infer_timestamp_column(weather)
        weather[tcol] = pd.to_datetime(weather[tcol])
        weather_df = weather[[tcol]].copy()
        weather_df = weather_df.rename(columns={tcol: 'timestamp'})
        
        # Extract temperature if available
        temp_cols = ['Temperature (°C)', 'Temperature', 'Temp', 'temp']
        for tc in temp_cols:
            if tc in weather.columns:
                weather_df['temperature'] = weather[tc].astype(float)
                break
        
        dfs.append(weather_df)

    if not dfs:
        return None

    # merge on timestamp
    df = dfs[0]
    for other in dfs[1:]:
        df = pd.merge(df, other, on='timestamp', how='outer')

    # ensure timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    # resample to cadence
    df = df.resample(cadence).mean()
    df = df.reset_index()
    df['house_id'] = int(house_id)
    return df


def build_full_dataframe(raw_root: Path, cadence: str = CADENCE):
    all_houses = []
    for house_dir in sorted(raw_root.glob('house_*')):
        # derive house id from folder name if possible
        try:
            hid = int(house_dir.name.split('_')[-1])
        except Exception:
            hid = len(all_houses) + 1
        df = read_house_folder(house_dir, hid, cadence=cadence)
        if df is None:
            continue
        all_houses.append(df)
    if not all_houses:
        raise SystemExit(f"No house data found under {raw_root}")
    full = pd.concat(all_houses, ignore_index=True)
    # drop rows without pv_kW (we predict pv)
    if 'pv_kW' not in full.columns:
        raise SystemExit('No pv_kW column found in concatenated house data')
    full = full.dropna(subset=['pv_kW'])
    return full


def prepare_timeseries_dataset(df: pd.DataFrame):
    # determine cadence minutes
    cadence_lower = CADENCE.lower()
    if cadence_lower.endswith('min'):
        cadence_min = int(cadence_lower.replace('min', ''))
    elif cadence_lower.endswith('h'):
        cadence_min = int(cadence_lower.replace('h', '')) * 60
    else:
        cadence_min = 15

    # filter to last N years (use all 3 years for training samples)
    max_ts = df['timestamp'].max()
    history_start = max_ts - pd.DateOffset(years=YEARS_OF_HISTORY)
    df = df[df['timestamp'] >= history_start].copy()
    if df.empty:
        raise SystemExit('No data after filtering to the requested history window')

    # Add time-based features for better accuracy
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # Cyclic encoding for hour and month (preserves circular nature)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # build global time_idx mapping
    unique_ts = np.sort(df['timestamp'].unique())
    ts_to_idx = {pd.Timestamp(t): i for i, t in enumerate(unique_ts)}
    df['time_idx'] = df['timestamp'].map(ts_to_idx)

    # Use optimal encoder length (14 days) instead of full 3 years
    encoder_steps = int((ENCODER_DAYS * 24 * 60) / cadence_min)
    pred_steps = int((PRED_DAYS * 24 * 60) / cadence_min)
    
    print(f"Encoder length: {ENCODER_DAYS} days ({encoder_steps} steps)")
    print(f"Prediction length: {PRED_DAYS} days ({pred_steps} steps)")

    # ensure categorical house id
    df['house_id'] = df['house_id'].astype(str)

    # handle missing values
    if 'load_kW' in df.columns:
        df['load_missing'] = df['load_kW'].isna().astype(int)
        df['load_kW'] = df['load_kW'].fillna(0.0)
    
    if 'temperature' in df.columns:
        df['temperature'] = df['temperature'].fillna(df['temperature'].mean())
    
    if 'irradiance' in df.columns:
        df['irradiance'] = df['irradiance'].fillna(0.0)

    # rolling features to preserve patterns with hourly cadence
    # windows: 3h and 24h
    win_3h = max(1, int(round(3 * 60 / cadence_min)))
    win_24h = max(1, int(round(24 * 60 / cadence_min)))
    df = df.sort_values(["house_id", "time_idx"]).copy()
    df['pv_roll3h'] = df.groupby('house_id')['pv_kW'].transform(lambda s: s.rolling(window=win_3h, min_periods=1).mean())
    df['pv_roll24h'] = df.groupby('house_id')['pv_kW'].transform(lambda s: s.rolling(window=win_24h, min_periods=1).mean())

    # Build lists of features for TimeSeriesDataSet
    time_varying_known_reals = ['time_idx', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    time_varying_unknown_reals = ['pv_kW', 'pv_roll3h', 'pv_roll24h']
    
    if 'load_kW' in df.columns:
        time_varying_unknown_reals.append('load_kW')
    if 'temperature' in df.columns:
        time_varying_unknown_reals.append('temperature')
    if 'irradiance' in df.columns:
        time_varying_unknown_reals.append('irradiance')
    
    time_varying_known_categoricals = []
    time_varying_unknown_categoricals = []
    static_categoricals = ['house_id']

    # create dataset
    training_cutoff = df['time_idx'].max() - pred_steps

    dataset = TimeSeriesDataSet(
        df,
        time_idx='time_idx',
        target='pv_kW',
        group_ids=['house_id'],
        max_encoder_length=encoder_steps,
        min_encoder_length=encoder_steps // 2,  # allow some flexibility
        max_prediction_length=pred_steps,
        static_categoricals=static_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=['house_id'], transformation='softplus'),
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # return both the dataset and the processed dataframe (with time_idx)
    return dataset, df


def main():
    print('Reading raw house data...')
    df = build_full_dataframe(RAW_IN, cadence=CADENCE)
    print('Total rows after concat:', len(df))
    ds, df = prepare_timeseries_dataset(df)

    # split by time: leave last prediction length for validation
    max_time_idx = df['time_idx'].max()
    pred_steps = int((PRED_DAYS * 24 * 60) / (int(CADENCE.replace('min','')) if 'min' in CADENCE else 60))
    training_cutoff = max_time_idx - pred_steps

    training = TimeSeriesDataSet(df[df['time_idx'] <= training_cutoff],
                                 time_idx='time_idx', target='pv_kW', group_ids=['house_id'],
                                 max_encoder_length=ds.max_encoder_length,
                                 min_encoder_length=ds.min_encoder_length,
                                 max_prediction_length=ds.max_prediction_length,
                                 static_categoricals=ds.static_categoricals,
                                 time_varying_known_reals=ds.time_varying_known_reals,
                                 time_varying_unknown_reals=ds.time_varying_unknown_reals,
                                 target_normalizer=ds.target_normalizer,
                                 allow_missing_timesteps=True,
                                 add_relative_time_idx=True,
                                 add_target_scales=True,
                                 add_encoder_length=True)

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

    # dataloaders
    train_dataloader = training.to_dataloader(
        train=True, batch_size=BATCH, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=BATCH, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True
    )

    # instantiate model from the training dataset with optimized hyperparameters
    from pytorch_forecasting.metrics import QuantileLoss
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEADS,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_SIZE // 2,
        output_size=7,  # predict quantiles for uncertainty estimation
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4,
    )
    
    print(f"\nModel Architecture:")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Attention heads: {ATTENTION_HEADS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Output: 7 quantiles (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)")

    # callbacks
    ckpt = ModelCheckpoint(dirpath=ART_DIR, filename='tft-{epoch}-{val_loss:.4f}', save_top_k=1, monitor='val_loss')
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min')
    lr_monitor = LearningRateMonitor()

    # prefer TensorBoard if available, otherwise fall back to CSVLogger
    try:
        logger = TensorBoardLogger(ART_DIR, name='tensorboard')
    except Exception:
        logger = CSVLogger(ART_DIR, name='csv_logs')

    trainer = Trainer(
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        max_epochs=1 if DRY_RUN else EPOCHS,
        callbacks=[ckpt, early_stop, lr_monitor],
        logger=logger,
        enable_checkpointing=True,
        gradient_clip_val=0.5,
    )

    print('Starting training (dry-run=%s)...' % DRY_RUN)
    trainer.fit(tft, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()