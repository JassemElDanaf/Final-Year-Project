# E:\FYP\src\agent1_infer_tomorrow.py
from pathlib import Path
import pandas as pd, numpy as np
import joblib

ROOT = Path(r"E:/FYP")
PROC = ROOT/"data/processed"
MODELS = ROOT/"models"

# load models
pv_m   = joblib.load(MODELS/"pv_nextday_xgb_gpu.pkl")
load_w = joblib.load(MODELS/"load_week_xgb_gpu.pkl")

# use the same feature builder as training
from src.features import hourly_with_features, add_lags_targets, time_split

# build features on all data (no split) and keep the latest day per house
hourly = hourly_with_features(limit_houses=10)  # or None for all
X_full = add_lags_targets(hourly).sort_values(["house_id","timestamp"])

# we just need the latest 24 rows per house (last day of feature time)
latest = X_full.groupby("house_id").tail(24).copy()

# select feature columns = everything non-id/non-target, same as training
drop_cols = {"timestamp","house_id","load_kw","pv_kw","dow","hour","month",
             "y_pv_nextday","y_load_nextweek"}
FEATS = [c for c in latest.columns if c not in drop_cols]

# predict
latest["pv_nextday_kw_pred"]    = pv_m.predict(latest[FEATS])
latest["load_nextweek_kw_pred"] = load_w.predict(latest[FEATS])

# align to target times
out = latest[["timestamp","house_id","pv_nextday_kw_pred","load_nextweek_kw_pred"]].copy()
out["pv_for_time"]   = out["timestamp"] + pd.Timedelta(hours=24)
out["load_for_time"] = out["timestamp"] + pd.Timedelta(hours=168)

# physical clipping
out["pv_nextday_kw_pred"]    = out["pv_nextday_kw_pred"].clip(lower=0)
out["load_nextweek_kw_pred"] = out["load_nextweek_kw_pred"].clip(lower=0)

# keep neat columns
out = out[["house_id","pv_for_time","pv_nextday_kw_pred","load_for_time","load_nextweek_kw_pred"]]
out.to_csv(PROC/"agent1_tomorrow_forecast.csv", index=False)
print("Saved →", PROC/"agent1_tomorrow_forecast.csv")
