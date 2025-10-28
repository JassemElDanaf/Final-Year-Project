from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(r"E:/FYP")
PROC = ROOT / "data/processed"
OUT_JSON = ROOT / "models/agent1_accuracy_summary.json"
OUT_CSV  = PROC / "agent1_accuracy_per_house.csv"

# ---------- metrics (NaN-safe helpers) ----------
def _clean_pairs(y, yhat, extra_mask=None):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if extra_mask is not None:
        mask &= np.asarray(extra_mask, dtype=bool)
    return y[mask], yhat[mask]

def mae(y, yhat, mask=None):
    y, yhat = _clean_pairs(y, yhat, mask)
    return float(mean_absolute_error(y, yhat)) if len(y) else float("nan")

def rmse(y, yhat, mask=None):
    y, yhat = _clean_pairs(y, yhat, mask)
    return float(np.sqrt(mean_squared_error(y, yhat))) if len(y) else float("nan")

def r2(y, yhat, mask=None):
    y, yhat = _clean_pairs(y, yhat, mask)
    return float(r2_score(y, yhat)) if len(y) else float("nan")

def mape_masked(y, yhat, min_actual, mask=None):
    y, yhat = _clean_pairs(y, yhat, mask)
    if not len(y): return float("nan")
    strong = np.abs(y) >= min_actual
    if not np.any(strong): return float("nan")
    return float(np.mean(np.abs((y[strong] - yhat[strong]) / np.clip(np.abs(y[strong]), 1e-6, None))) * 100)

def smape(y, yhat, mask=None):
    y, yhat = _clean_pairs(y, yhat, mask)
    if not len(y): return float("nan")
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    strong = denom > 1e-6
    if not np.any(strong): return float("nan")
    return float(np.mean(np.abs(y[strong] - yhat[strong]) / denom[strong]) * 100)

# ---------- load artifacts ----------
pred = pd.read_csv(PROC/"agent1_predictions_aligned.csv",
                   parse_dates=["pv_for_time","load_for_time"])
raw = pd.read_csv(PROC/"all_houses_15min.csv", parse_dates=["timestamp"])

hourly = (raw.set_index("timestamp")
            .groupby("house_id")
            .resample("1h")
            .agg({"pv_kw":"mean","load_kw":"mean"})
            .reset_index())

# ----- PV join & baseline (t-24h) -----
pv = pred[["house_id","pv_for_time","pv_nextday_kw_pred"]].rename(
    columns={"pv_for_time":"target_time","pv_nextday_kw_pred":"pv_pred"}
).merge(
    hourly.rename(columns={"timestamp":"target_time"}),
    on=["house_id","target_time"], how="left"
).rename(columns={"pv_kw":"pv_actual"})

ref = hourly.copy()
ref["target_time"] = ref["timestamp"] + pd.Timedelta(hours=24)
pv = pv.merge(ref[["house_id","target_time","pv_kw"]]
              .rename(columns={"pv_kw":"pv_baseline"}),
              on=["house_id","target_time"], how="left")

PV_MIN = 0.05  # kW: evaluate only daylight-ish actuals
pv_mask = pv["pv_actual"] >= PV_MIN

# ----- Load join & baseline (t-168h) -----
ld = pred[["house_id","load_for_time","load_nextweek_kw_pred"]].rename(
    columns={"load_for_time":"target_time","load_nextweek_kw_pred":"load_pred"}
).merge(
    hourly.rename(columns={"timestamp":"target_time"}),
    on=["house_id","target_time"], how="left"
).rename(columns={"load_kw":"load_actual"})

refL = hourly.copy()
refL["target_time"] = refL["timestamp"] + pd.Timedelta(hours=168)
ld = ld.merge(refL[["house_id","target_time","load_kw"]]
              .rename(columns={"load_kw":"load_baseline"}),
              on=["house_id","target_time"], how="left")

LOAD_MIN = 0.10  # kW: ignore near-idle
ld_mask = ld["load_actual"] >= LOAD_MIN

# ---------- global metrics (NaN-safe) ----------
summary = {
    # PV
    "PV_R2":            r2(pv["pv_actual"], pv["pv_pred"], pv_mask),
    "PV_MAE_kW":        mae(pv["pv_actual"], pv["pv_pred"], pv_mask),
    "PV_RMSE_kW":       rmse(pv["pv_actual"], pv["pv_pred"], pv_mask),
    "PV_MAPE_%":        mape_masked(pv["pv_actual"], pv["pv_pred"], PV_MIN, pv_mask),
    "PV_sMAPE_%":       smape(pv["pv_actual"], pv["pv_pred"], pv_mask),
    "PV_base_R2":       r2(pv["pv_actual"], pv["pv_baseline"], pv_mask),
    "PV_base_MAE_kW":   mae(pv["pv_actual"], pv["pv_baseline"], pv_mask),
    "PV_base_RMSE_kW":  rmse(pv["pv_actual"], pv["pv_baseline"], pv_mask),
    "PV_base_MAPE_%":   mape_masked(pv["pv_actual"], pv["pv_baseline"], PV_MIN, pv_mask),
    "PV_base_sMAPE_%":  smape(pv["pv_actual"], pv["pv_baseline"], pv_mask),
    "PV_points_used":   int(np.sum(np.isfinite(pv["pv_actual"]) & np.isfinite(pv["pv_pred"]) & pv_mask)),
    # Load
    "Load_R2":            r2(ld["load_actual"], ld["load_pred"], ld_mask),
    "Load_MAE_kW":        mae(ld["load_actual"], ld["load_pred"], ld_mask),
    "Load_RMSE_kW":       rmse(ld["load_actual"], ld["load_pred"], ld_mask),
    "Load_MAPE_%":        mape_masked(ld["load_actual"], ld["load_pred"], LOAD_MIN, ld_mask),
    "Load_sMAPE_%":       smape(ld["load_actual"], ld["load_pred"], ld_mask),
    "Load_base_R2":       r2(ld["load_actual"], ld["load_baseline"], ld_mask),
    "Load_base_MAE_kW":   mae(ld["load_actual"], ld["load_baseline"], ld_mask),
    "Load_base_RMSE_kW":  rmse(ld["load_actual"], ld["load_baseline"], ld_mask),
    "Load_base_MAPE_%":   mape_masked(ld["load_actual"], ld["load_baseline"], LOAD_MIN, ld_mask),
    "Load_base_sMAPE_%":  smape(ld["load_actual"], ld["load_baseline"], ld_mask),
    "Load_points_used":   int(np.sum(np.isfinite(ld["load_actual"]) & np.isfinite(ld["load_pred"]) & ld_mask)),
}

# ---------- per-house table ----------
rows = []
for hid, g in pv[pv_mask].groupby("house_id"):
    rows.append({
        "house_id": hid,
        "pv_R2": r2(g["pv_actual"], g["pv_pred"]),
        "pv_MAE": mae(g["pv_actual"], g["pv_pred"]),
        "pv_base_MAE": mae(g["pv_actual"], g["pv_baseline"])
    })
for hid, g in ld[ld_mask].groupby("house_id"):
    r = next((x for x in rows if x["house_id"]==hid), {"house_id": hid})
    r.update({
        "load_R2": r2(g["load_actual"], g["load_pred"]),
        "load_MAE": mae(g["load_actual"], g["load_pred"]),
        "load_base_MAE": mae(g["load_actual"], g["load_baseline"])
    })
    if r not in rows: rows.append(r)

per_house = pd.DataFrame(rows).sort_values("house_id")
per_house.to_csv(OUT_CSV, index=False)

# ---------- save + print ----------
OUT_JSON.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("Saved per-house →", OUT_CSV)
print("Saved summary  →", OUT_JSON)
