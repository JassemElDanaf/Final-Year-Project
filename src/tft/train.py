# -*- coding: utf-8 -*-
"""
Train a Temporal Fusion Transformer (TFT) on multi-home PV/load data.
- Trains on ALL days (not only Mondays)
- Predicts multi-horizon (e.g., next-day blocks); we can later slice “next-week same day”
Artifacts:
- Best checkpoint -> models/tft/best.ckpt
"""

from pathlib import Path
import numpy as np
import pandas as pd

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

# ----------------- CONFIG -----------------
ROOT = Path(__file__).resolve().parents[2]      # repo root (…/FYP)
DATA_IN  = ROOT / "data" / "processed"         # input processed tables
ART_DIR  = ROOT / "models" / "tft"             # checkpoints
BATCH    = 128
PRECISION = 16       # fp16 is friendly for GTX 1650
EPOCHS    = 50
ENC_DAYS  = 5        # encoder/context length in days
PRED_DAYS = 1        # prediction length in days
CADENCE   = "15min"  # set to "10min" if that is your grid

# ----------------- LOAD DATA -----------------
# Load the CSV file we created
print("Loading data...")
df = pd.read_csv(DATA_IN / "multi_home_15min.csv")
print(f"Data shape: {df.shape}")

# ------- choose TARGET now (PV generation) -------
print("Processing target...")
df["target"] = df["pv_kw"].astype("float32")  # Using pv_kw as our target
df = df.dropna(subset=["target"])  # Remove rows with missing target values
print(f"Shape after dropping NA: {df.shape}")

# ----------------- FEATURES -----------------
# integer time index on a fixed cadence
step_minutes = 15 if CADENCE == "15min" else 10
df["time_idx"] = (pd.to_datetime(df["timestamp"]).view("int64") // (step_minutes*60*1e9)).astype("int64")

dt = pd.to_datetime(df["timestamp"])
df["hour"] = dt.dt.hour
df["dow"]  = dt.dt.weekday
df["is_weekend"] = (df["dow"] >= 5).astype("int8")
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24).astype("float32")
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24).astype("float32")

# ----------------- SPLIT (strict by time) -----------------
tmin, tmax = df["time_idx"].min(), df["time_idx"].max()
train_cut  = int(tmin + 0.80*(tmax - tmin))
val_cut    = int(tmin + 0.90*(tmax - tmin))
train_df = df[df.time_idx <= train_cut].copy()
val_df   = df[(df.time_idx > train_cut) & (df.time_idx <= val_cut)].copy()
test_df  = df[df.time_idx > val_cut].copy()

# ----------------- WINDOW LENGTHS -----------------
steps_per_day = int((24*60)//step_minutes)
max_encoder_length    = ENC_DAYS  * steps_per_day
max_prediction_length = PRED_DAYS * steps_per_day

# ----------------- DATASET DEF -----------------
# TimeSeriesDataSet wires static IDs, known-future & unknown reals, and per-group scaling. (Docs)  # noqa
# https://pytorch-forecasting.readthedocs.io/en/v1.4.0/api/pytorch_forecasting.data.timeseries._timeseries.TimeSeriesDataSet.html
training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="target",
    group_ids=["house_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    target_normalizer=GroupNormalizer(groups=["house_id"], transformation="standard"),  # per-house scale  # noqa
    static_categoricals=["house_id"],
    time_varying_known_reals=["hour_sin", "hour_cos", "dow", "is_weekend"],            # known in future
    time_varying_unknown_reals=["target","pv","temp","irradiance","wind"],             # observed/past vars
)

validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)
testing    = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)

train_loader = training.to_dataloader(train=True,  batch_size=BATCH, num_workers=4)
val_loader   = validation.to_dataloader(train=False, batch_size=BATCH, num_workers=4)
test_loader  = testing.to_dataloader(train=False,  batch_size=BATCH, num_workers=4)

# ----------------- MODEL -----------------
# TFT is built for multi-horizon forecasts with static IDs + known-future/observed features (paper + docs).  # noqa
# https://arxiv.org/abs/1912.09363  |  https://pytorch-forecasting.readthedocs.io/en/v1.4.0/
tft = TemporalFusionTransformer.from_dataset(
    training,
    hidden_size=64,                  # scale for 4GB GPU
    attention_head_size=4,
    dropout=0.1,
    learning_rate=1e-3,
    optimizer="adam",
    loss="mae",                      # point forecast; switch to QuantileLoss() for P50/P90 bands
    reduce_on_plateau_patience=4,
)

ART_DIR.mkdir(parents=True, exist_ok=True)
early = EarlyStopping(monitor="val_loss", patience=10, mode="min")  # stop when no val improvement   # noqa
# Lightning callbacks docs: EarlyStopping & ModelCheckpoint                                          
# https://lightning.ai/docs/pytorch/stable/common/early_stopping.html                                
# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html      
ckpt  = ModelCheckpoint(dirpath=str(ART_DIR), monitor="val_loss", mode="min", save_top_k=1, filename="best")

trainer = Trainer(
    max_epochs=EPOCHS,
    accelerator="gpu", devices=1, precision=PRECISION,
    callbacks=[early, ckpt],
    log_every_n_steps=50
)

seed_everything(42)
trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

print(f"Best checkpoint: {ckpt.best_model_path}")
# Optional quick smoke test on test set (metrics will be computed later in your metrics module)
# predictions = TemporalFusionTransformer.load_from_checkpoint(ckpt.best_model_path).predict(test_loader)
