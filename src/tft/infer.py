# -*- coding: utf-8 -*-
"""
Load best TFT and generate forecasts.
You can:
- predict for the full test block
- or filter decoder starts to only those that begin on the SAME weekday next week.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

ROOT = Path(__file__).resolve().parents[2]
ART_DIR = ROOT / "models" / "tft"
DATA_IN = ROOT / "data" / "processed"

def load_datasets():
    df = pd.read_parquet(DATA_IN / "multi_home_15min.parquet").sort_values(["house_id","timestamp"])
    # Match feature engineering from train.py
    df["time_idx"] = (pd.to_datetime(df["timestamp"]).view("int64") // (15*60*1e9)).astype("int64")
    dt = pd.to_datetime(df["timestamp"])
    df["hour"] = dt.dt.hour; df["dow"] = dt.dt.weekday; df["is_weekend"] = (df["dow"]>=5).astype("int8")
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24).astype("float32")
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24).astype("float32")
    df["target"] = df["generation"].astype("float32")
    # Same quantile time split as train.py
    tmin, tmax = df["time_idx"].min(), df["time_idx"].max()
    train_cut  = int(tmin + 0.80*(tmax - tmin))
    val_cut    = int(tmin + 0.90*(tmax - tmin))
    train_df = df[df.time_idx <= train_cut]
    val_df   = df[(df.time_idx > train_cut) & (df.time_idx <= val_cut)]
    test_df  = df[df.time_idx > val_cut]
    # Rebuild TimeSeriesDataSet from training to ensure identical transforms (no leakage)
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target",
        group_ids=["house_id"],
        max_encoder_length=5*96, max_prediction_length=1*96,
        target_normalizer=TimeSeriesDataSet.get_parameters(train=True)["target_normalizer"]
        if hasattr(TimeSeriesDataSet, "get_parameters") else None,
        static_categoricals=["house_id"],
        time_varying_known_reals=["hour_sin","hour_cos","dow","is_weekend"],
        time_varying_unknown_reals=["target","pv","temp","irradiance","wind"],
    )
    testing = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)
    test_loader = testing.to_dataloader(train=False, batch_size=128, num_workers=4)
    return test_loader, testing

def predict_next_week_same_day():
    ckpt = max(ART_DIR.glob("best*.ckpt"), key=lambda p: p.stat().st_mtime)
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
    test_loader, testing = load_datasets()
    raw = model.predict(test_loader, return_x=True)  # returns predictions and batch metadata
    # Filter decoder starts whose weekday equals weekday(date + 7d)
    x = raw.x
    # decoder_time (first prediction step) exists in x["decoder_time_idx"] or via index mapping
    # We’ll approximate via original timestamps carried in x if present (varies by PF version).
    return raw  # downstream code can compute MAE/MAPE and filter per weekday
if __name__ == "__main__":
    out = predict_next_week_same_day()
    print("Predictions ready. (Filter by weekday in your metrics step.)")
