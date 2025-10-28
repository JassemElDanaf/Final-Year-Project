import pandas as pd, numpy as np
from pathlib import Path
from .config import PROC

EXOG = ["ghi", "temp_c", "rh", "wind_ms", "cloud_pct", "price_eur_kwh"]

def hourly_with_features(limit_houses=10):
    df = pd.read_csv(PROC / "all_houses_15min.csv", parse_dates=["timestamp"])

    if limit_houses:
        df = df[df["house_id"].isin(range(1, limit_houses + 1))].copy()

    # Aggregate to hourly means (new alias '1h')
    hourly = (
        df.set_index("timestamp")
          .groupby("house_id")
          .resample("1h")
          .agg({
              "load_kw": "mean", "pv_kw": "mean",
              "ghi": "mean", "temp_c": "mean", "rh": "mean",
              "wind_ms": "mean", "cloud_pct": "mean",
              "price_eur_kwh": "mean",
              "dow": "first", "hour": "first", "month": "first",
          })
          .reset_index()
    )

    # Light imputation for exogenous features (so lagging doesn't kill all rows)
    for col in EXOG:
        if col in hourly.columns:
            hourly[col] = (
                hourly.groupby("house_id")[col]
                      .transform(lambda s: s.ffill().bfill())
            )

    return hourly

def _add_lags(g: pd.DataFrame, cols, lags):
    for c in cols:
        if c not in g.columns:  # skip missing columns gracefully
            continue
        for L in lags:
            g[f"{c}_lag{L}"] = g[c].shift(L)
    return g

def add_lags_targets(hourly: pd.DataFrame) -> pd.DataFrame:
    # Build per-house to avoid groupby.apply warnings and keep full control
    frames = []
    for hid, g in hourly.sort_values(["house_id", "timestamp"]).groupby("house_id", sort=False):
        g = _add_lags(
            g.copy(),
            cols=["load_kw", "pv_kw"] + EXOG,
            lags=(1, 2, 24, 48, 24*7, 24*14)
        )
        # Targets
        g["y_pv_nextday"]    = g["pv_kw"].shift(-24)
        g["y_load_nextweek"] = g["load_kw"].shift(-24*7)
        frames.append(g)

    X = pd.concat(frames, ignore_index=True)

    # Drop rows that don't have targets or essential lags
    essential = ["y_pv_nextday", "y_load_nextweek",
                 "load_kw_lag24", "pv_kw_lag24"]
    exist_essential = [c for c in essential if c in X.columns]
    X = X.dropna(subset=exist_essential).reset_index(drop=True)

    return X

def time_split(X: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    if X.empty:
        raise ValueError("Feature table is empty after lagging/imputation. "
                         "Check that processed data exists and has non-null values.")

    tmin, tmax = X["timestamp"].min(), X["timestamp"].max()
    s1 = tmin + (tmax - tmin) * train_frac
    s2 = tmin + (tmax - tmin) * (train_frac + val_frac)

    train = X[X["timestamp"] <= s1]
    val   = X[(X["timestamp"] > s1) & (X["timestamp"] <= s2)]
    test  = X[X["timestamp"] > s2]

    # Features = everything except identifiers and targets
    drop_cols = {"timestamp", "house_id", "load_kw", "pv_kw", "dow", "hour", "month",
                 "y_pv_nextday", "y_load_nextweek"}
    FEATS = [c for c in X.columns if c not in drop_cols]

    if train.empty or not FEATS:
        raise ValueError(f"Train split or feature set empty. "
                         f"len(train)={len(train)}, n_features={len(FEATS)}")

    return train, val, test, FEATS
import pandas as pd, numpy as np
from pathlib import Path
from .config import PROC

EXOG = ["ghi", "temp_c", "rh", "wind_ms", "cloud_pct", "price_eur_kwh"]

def hourly_with_features(limit_houses=10):
    df = pd.read_csv(PROC / "all_houses_15min.csv", parse_dates=["timestamp"])

    if limit_houses:
        df = df[df["house_id"].isin(range(1, limit_houses + 1))].copy()

    # Aggregate to hourly means (new alias '1h')
    hourly = (
        df.set_index("timestamp")
          .groupby("house_id")
          .resample("1h")
          .agg({
              "load_kw": "mean", "pv_kw": "mean",
              "ghi": "mean", "temp_c": "mean", "rh": "mean",
              "wind_ms": "mean", "cloud_pct": "mean",
              "price_eur_kwh": "mean",
              "dow": "first", "hour": "first", "month": "first",
          })
          .reset_index()
    )

    # Light imputation for exogenous features (so lagging doesn't kill all rows)
    for col in EXOG:
        if col in hourly.columns:
            hourly[col] = (
                hourly.groupby("house_id")[col]
                      .transform(lambda s: s.ffill().bfill())
            )

    return hourly

def _add_lags(g: pd.DataFrame, cols, lags):
    for c in cols:
        if c not in g.columns:  # skip missing columns gracefully
            continue
        for L in lags:
            g[f"{c}_lag{L}"] = g[c].shift(L)
    return g

def add_lags_targets(hourly: pd.DataFrame) -> pd.DataFrame:
    # Build per-house to avoid groupby.apply warnings and keep full control
    frames = []
    for hid, g in hourly.sort_values(["house_id", "timestamp"]).groupby("house_id", sort=False):
        g = _add_lags(
            g.copy(),
            cols=["load_kw", "pv_kw"] + EXOG,
            lags=(1, 2, 24, 48, 24*7, 24*14)
        )
        # Targets
        g["y_pv_nextday"]    = g["pv_kw"].shift(-24)
        g["y_load_nextweek"] = g["load_kw"].shift(-24*7)
        frames.append(g)

    X = pd.concat(frames, ignore_index=True)

    # Drop rows that don't have targets or essential lags
    essential = ["y_pv_nextday", "y_load_nextweek",
                 "load_kw_lag24", "pv_kw_lag24"]
    exist_essential = [c for c in essential if c in X.columns]
    X = X.dropna(subset=exist_essential).reset_index(drop=True)

    return X

def time_split(X: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    if X.empty:
        raise ValueError("Feature table is empty after lagging/imputation. "
                         "Check that processed data exists and has non-null values.")

    tmin, tmax = X["timestamp"].min(), X["timestamp"].max()
    s1 = tmin + (tmax - tmin) * train_frac
    s2 = tmin + (tmax - tmin) * (train_frac + val_frac)

    train = X[X["timestamp"] <= s1]
    val   = X[(X["timestamp"] > s1) & (X["timestamp"] <= s2)]
    test  = X[X["timestamp"] > s2]

    # Features = everything except identifiers and targets
    drop_cols = {"timestamp", "house_id", "load_kw", "pv_kw", "dow", "hour", "month",
                 "y_pv_nextday", "y_load_nextweek"}
    FEATS = [c for c in X.columns if c not in drop_cols]

    if train.empty or not FEATS:
        raise ValueError(f"Train split or feature set empty. "
                         f"len(train)={len(train)}, n_features={len(FEATS)}")

    return train, val, test, FEATS
