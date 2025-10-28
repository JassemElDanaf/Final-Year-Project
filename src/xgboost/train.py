import argparse, json
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

from .data import build_processed_for_all
from .features import hourly_with_features, add_lags_targets, time_split
from .config import PROC, MODELS

# --- metrics ---
from sklearn.metrics import mean_absolute_error
def mae(y, yhat):  return float(mean_absolute_error(y, yhat))
def mape(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.mean(np.abs((y - yhat) / np.clip(np.abs(y), 1e-6, None))) * 100)

def to_float32(df):
    out = df.copy()
    for c in out.columns:
        if np.issubdtype(out[c].dtype, np.number):
            out[c] = out[c].astype(np.float32)
    return out

def main(limit_houses=10):
    # 1) Build processed if missing
    try:
        pd.read_csv(PROC / "all_houses_15min.csv", nrows=1)
    except FileNotFoundError:
        build_processed_for_all()

    # 2) Features + splits
    hourly = hourly_with_features(limit_houses=limit_houses)
    X = add_lags_targets(hourly)
    train, val, test, FEATS = time_split(X)

    # cast to float32 for GPU efficiency
    X_train, X_val, X_test = to_float32(train[FEATS]), to_float32(val[FEATS]), to_float32(test[FEATS])
    y_pv_tr, y_pv_val, y_pv_te = train["y_pv_nextday"].values,   val["y_pv_nextday"].values,   test["y_pv_nextday"].values
    y_ld_tr, y_ld_val, y_ld_te = train["y_load_nextweek"].values, val["y_load_nextweek"].values, test["y_load_nextweek"].values

    print(f"[Agent1-GPU] shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}, feats={len(FEATS)}")

    # 3) GPU models (works on XGBoost 1.x and 2.x)
    common = dict(
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        n_estimators=800,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        max_depth=8,
        random_state=42,
        n_jobs=0,
        reg_lambda=1.0,
        eval_metric="rmse"   # put in constructor for 2.x compatibility
    )
    pv_m   = XGBRegressor(**common)
    load_w = XGBRegressor(**common)

    # 4) Fit (no early stopping; eval_set just for monitoring)
    pv_m.fit(X_train, y_pv_tr, eval_set=[(X_val, y_pv_val)])
    load_w.fit(X_train, y_ld_tr, eval_set=[(X_val, y_ld_val)])

    # 5) Metrics
    pv_pred   = pv_m.predict(X_test)
    load_pred = load_w.predict(X_test)
    metrics = {
        "pv_nextday":  {"MAE_kW": mae(y_pv_te, pv_pred),   "MAPE_%": mape(y_pv_te, pv_pred)},
        "load_week":   {"MAE_kW": mae(y_ld_te, load_pred), "MAPE_%": mape(y_ld_te, load_pred)},
        "n_train": len(train), "n_val": len(val), "n_test": len(test), "n_feats": len(FEATS),
        "gpu": "xgboost gpu_hist (no early stopping)"
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / "agent1_metrics.json").write_text(json.dumps(metrics, indent=2))

    # 6) Save predictions
    out = test[["timestamp","house_id"]].copy()
    out["pv_nextday_kw_pred"]     = pv_pred
    out["load_nextweek_kw_pred"]  = load_pred
    out.to_csv(PROC / "agent1_predictions.csv", index=False)

    # 7) Save models
    joblib.dump(pv_m,   MODELS / "pv_nextday_xgb_gpu.pkl")
    joblib.dump(load_w, MODELS / "load_week_xgb_gpu.pkl")

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions → {PROC/'agent1_predictions.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", type=int, default=10)
    args = ap.parse_args()
    main(limit_houses=args.houses)
