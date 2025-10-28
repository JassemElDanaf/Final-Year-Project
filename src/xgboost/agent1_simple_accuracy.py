from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(r"E:/FYP")
PROC = ROOT/"data/processed"

# load predictions aligned to target times
pred = pd.read_csv(PROC/"agent1_predictions_aligned.csv",
                   parse_dates=["pv_for_time","load_for_time"])
# build hourly actuals
raw = pd.read_csv(PROC/"all_houses_15min.csv", parse_dates=["timestamp"])
hourly = (raw.set_index("timestamp")
            .groupby("house_id").resample("1h")
            .agg({"pv_kw":"mean","load_kw":"mean"}).reset_index())

# join PV
pv = (pred[["house_id","pv_for_time","pv_nextday_kw_pred"]]
      .rename(columns={"pv_for_time":"target_time","pv_nextday_kw_pred":"pred"})
      .merge(hourly.rename(columns={"timestamp":"target_time","pv_kw":"actual"}),
             on=["house_id","target_time"], how="left"))
# evaluate only daylight (avoid div by ~0)
pv = pv[pv["actual"] >= 0.05].dropna(subset=["actual","pred"])

# join Load
ld = (pred[["house_id","load_for_time","load_nextweek_kw_pred"]]
      .rename(columns={"load_for_time":"target_time","load_nextweek_kw_pred":"pred"})
      .merge(hourly.rename(columns={"timestamp":"target_time","load_kw":"actual"}),
             on=["house_id","target_time"], how="left"))
# ignore near-idle loads
ld = ld[ld["actual"] >= 0.10].dropna(subset=["actual","pred"])

def smape(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    denom = (np.abs(y) + np.abs(yhat))/2.0
    m = denom > 1e-6
    return float(np.mean(np.abs(y[m]-yhat[m]) / denom[m]) * 100)

def hit_rate(y, yhat, tol):
    err = np.abs(np.asarray(y) - np.asarray(yhat))
    return float((err <= tol).mean() * 100)

# ---- PV accuracy ----
pv_smape = smape(pv["actual"], pv["pred"])
pv_acc   = 100 - pv_smape                     # sMAPE accuracy (%)
pv_hit   = hit_rate(pv["actual"], pv["pred"], tol=0.20)  # % within 0.20 kW

# ---- Load accuracy ----
ld_smape = smape(ld["actual"], ld["pred"])
ld_acc   = 100 - ld_smape
ld_hit   = hit_rate(ld["actual"], ld["pred"], tol=0.30)  # adjust tol if you like

out = {
    "PV_accuracy_% (100-sMAPE)": round(pv_acc, 2),
    "PV_hit_rate_% (|err|<=0.20kW)": round(pv_hit, 1),
    "Load_accuracy_% (100-sMAPE)": round(ld_acc, 2),
    "Load_hit_rate_% (|err|<=0.30kW)": round(ld_hit, 1),
    "PV_points": int(len(pv)), "Load_points": int(len(ld))
}
import json; print(json.dumps(out, indent=2))
