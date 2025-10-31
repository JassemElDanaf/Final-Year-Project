# -*- coding: utf-8 -*-
"""
Create an Agent1-style tomorrow forecast using the trained TFT model.

This script loads the best TFT checkpoint (saved by `src/tft/train.py`), builds
per-house encoder inputs from the processed 15min table and predicts the next-day
generation horizon (1 day). Output matches the xgboost `agent1_tomorrow_forecast.csv`
format for PV: one row per house per hour with column `pv_nextday_kw_pred`.

Notes:
- TFT was trained at 15-min cadence; we average 4 timesteps to obtain hourly
  predictions (to match the xgboost hourly output).
- `load_nextweek_kw_pred` is left as NaN here. If you want TFT to also predict
  weekly load you can train a second TFT with PRED_DAYS=7 (separate model).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

ROOT = Path(__file__).resolve().parents[2]
ART_DIR = ROOT / "models" / "tft"
DATA_IN = ROOT / "data" / "processed"
OUT = DATA_IN / "agent1_tomorrow_forecast_tft.csv"


def build_dataset_params():
    # mirror train.py settings
    ENC_DAYS = 5
    PRED_DAYS = 1
    step_minutes = 15
    steps_per_day = int((24 * 60) // step_minutes)
    max_encoder_length = ENC_DAYS * steps_per_day
    max_prediction_length = PRED_DAYS * steps_per_day
    return max_encoder_length, max_prediction_length


def load_data():
    df = pd.read_parquet(DATA_IN / "multi_home_15min.parquet").sort_values(["house_id", "timestamp"])  # noqa
    df = df.copy()
    # features as in train.py
    step_minutes = 15
    df["time_idx"] = (pd.to_datetime(df["timestamp"]).view("int64") // (step_minutes * 60 * 1e9)).astype("int64")
    dt = pd.to_datetime(df["timestamp"])
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.weekday
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")
    df["target"] = df["generation"].astype("float32")
    return df


def build_training_dataset(df, max_encoder_length, max_prediction_length):
    # Recreate the TimeSeriesDataSet used for training so transforms are identical
    tmin, tmax = df["time_idx"].min(), df["time_idx"].max()
    train_cut = int(tmin + 0.80 * (tmax - tmin))
    train_df = df[df.time_idx <= train_cut].copy()

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target",
        group_ids=["house_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        target_normalizer=None,
        static_categoricals=["house_id"],
        time_varying_known_reals=["hour_sin", "hour_cos", "dow", "is_weekend"],
        time_varying_unknown_reals=["target", "pv", "temp", "irradiance", "wind"],
    )
    return training


def make_per_house_prediction(model, training, df, max_encoder_length, max_prediction_length):
    houses = sorted(df["house_id"].unique())
    rows = []
    for hid in houses:
        g = df[df["house_id"] == hid].sort_values("time_idx").copy()
        if g.empty:
            continue
        # take last encoder window
        enc = g.tail(max_encoder_length).copy()
        last_ts = pd.to_datetime(enc["timestamp"].iloc[-1])

        # build decoder rows placeholders for the prediction horizon
        # create timestamps at 15-min cadence for the next day
        periods = max_prediction_length
        dec_times = pd.date_range(start=last_ts + pd.Timedelta(minutes=15), periods=periods, freq=f"{15}min")
        dec_time_idx = (dec_times.view("int64") // (15 * 60 * 1e9)).astype("int64")

        dec = pd.DataFrame({
            "timestamp": dec_times,
            "house_id": hid,
            "time_idx": dec_time_idx,
        })
        # compute known future features
        dt = pd.to_datetime(dec["timestamp"])
        dec["hour"] = dt.dt.hour
        dec["dow"] = dt.dt.weekday
        dec["is_weekend"] = (dec["dow"] >= 5).astype("int8")
        dec["hour_sin"] = np.sin(2 * np.pi * dec["hour"] / 24).astype("float32")
        dec["hour_cos"] = np.cos(2 * np.pi * dec["hour"] / 24).astype("float32")
        # unknown reals can be NaN for prediction
        for c in ["target", "pv", "temp", "irradiance", "wind"]:
            dec[c] = np.nan

        # Concatenate encoder + decoder to form a single example for this house
        example = pd.concat([enc, dec], ignore_index=True, sort=False)

        # Build dataset for this single example
        ds = TimeSeriesDataSet.from_dataset(training, example, predict=True, stop_randomization=True)
        dl = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
        preds = model.predict(dl)  # shape: (1, prediction_length)
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(1, -1)
        preds = preds.squeeze(0)

        # map preds (15-min) to hourly by averaging every 4 steps
        # build timestamps for hourly bins
        hourly_times = [dec_times[i] for i in range(0, periods, 4)]
        hourly_preds = []
        for i in range(0, periods, 4):
            seg = preds[i:i+4]
            hourly_preds.append(np.nanmean(seg))

        for ts, p in zip(hourly_times, hourly_preds):
            rows.append({
                "house_id": hid,
                "pv_for_time": ts,
                "pv_nextday_kw_pred": float(np.clip(p, 0, None)) if not np.isnan(p) else np.nan,
                "load_for_time": pd.NaT,
                "load_nextweek_kw_pred": np.nan,
            })

    out = pd.DataFrame(rows)
    # reorder & save
    out = out[["house_id", "pv_for_time", "pv_nextday_kw_pred", "load_for_time", "load_nextweek_kw_pred"]]
    out.to_csv(OUT, index=False)
    print("Saved →", OUT)


def main():
    max_encoder_length, max_prediction_length = build_dataset_params()
    df = load_data()
    training = build_training_dataset(df, max_encoder_length, max_prediction_length)

    ckpt = max(ART_DIR.glob("best*.ckpt"), key=lambda p: p.stat().st_mtime)
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
    model.eval()

    make_per_house_prediction(model, training, df, max_encoder_length, max_prediction_length)


if __name__ == "__main__":
    main()
