# -*- coding: utf-8 -*-
"""
Quick-start script to train a basic Temporal Fusion Transformer on CPU.

This script uses the existing processed `multi_home_15min.csv` dataset and
runs a short training loop (few epochs) so you can validate the full TFT
pipeline on CPU before moving to GPU.

Usage:
    python src/tft/train_basic_cpu.py

The script writes a checkpoint to `models/tft/basic_cpu.ckpt`.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE


# ----------------- CONFIG -----------------
ROOT = Path(__file__).resolve().parents[2]
DATA_IN = ROOT / "data" / "processed"
ART_DIR = ROOT / "models" / "tft"
ART_DIR.mkdir(parents=True, exist_ok=True)

# Modest settings for CPU-run validation
import os

BATCH = 64
# Allow quick override for testing: set CPU_EPOCHS environment variable to an int
EPOCHS = int(os.getenv("CPU_EPOCHS", "8"))
ENC_DAYS = 3
PRED_DAYS = 1
CADENCE = "15min"
SEED = 42

seed_everything(SEED)


def main():
    print("Loading processed data...")
    df = pd.read_csv(DATA_IN / "multi_home_15min.csv")
    print(f"Data shape: {df.shape}")

    # target and basic preprocessing
    df["target"] = df["pv_kw"].astype("float32")
    # ensure house_id is categorical (string) for TimeSeriesDataSet
    if "house_id" in df.columns:
        df["house_id"] = df["house_id"].astype(str)
    df = df.dropna(subset=["target"]).copy()

    step_minutes = 15 if CADENCE == "15min" else 10
    df["time_idx"] = (
        (pd.to_datetime(df["timestamp"]).view("int64") // (step_minutes * 60 * 1e9)).astype("int64")
    )
    dt = pd.to_datetime(df["timestamp"])
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.weekday
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")

    # Time splits (80/10/10 by time idx)
    tmin, tmax = df["time_idx"].min(), df["time_idx"].max()
    train_cut = int(tmin + 0.80 * (tmax - tmin))
    val_cut = int(tmin + 0.90 * (tmax - tmin))
    train_df = df[df.time_idx <= train_cut].copy()
    val_df = df[(df.time_idx > train_cut) & (df.time_idx <= val_cut)].copy()
    test_df = df[df.time_idx > val_cut].copy()

    steps_per_day = int((24 * 60) // step_minutes)
    max_encoder_length = ENC_DAYS * steps_per_day
    max_prediction_length = PRED_DAYS * steps_per_day

    # dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target",
        group_ids=["house_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        target_normalizer=GroupNormalizer(groups=["house_id"]),
        static_categoricals=["house_id"],
        time_varying_known_reals=["hour_sin", "hour_cos", "dow", "is_weekend"],
        # match column names in processed data: temp_c, ghi (irradiance), wind_ms
        time_varying_unknown_reals=["target", "pv_kw", "temp_c", "ghi", "wind_ms"],
        # allow gaps in the processed time series (missing timesteps)
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)
    testing = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)

    train_loader = training.to_dataloader(train=True, batch_size=BATCH, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=BATCH, num_workers=0)

    # model - basic TFT configuration (not huge, but uses attention and gating)
    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=128,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=64,
        learning_rate=1e-3,
        optimizer="adam",
    loss=MAE(),
    )

    # callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=ART_DIR, filename="basic_cpu", monitor="val_loss", save_top_k=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        enable_progress_bar=True,
    )

    print("Starting training (CPU)...")
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Training finished. Best model saved to:", checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
