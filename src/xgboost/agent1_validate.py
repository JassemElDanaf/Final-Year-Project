from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

ROOT = Path(r"E:/FYP")
PROC = ROOT / "data/processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Load predictions (already aligned to target times)
pred = pd.read_csv(
    PROC / "agent1_predictions_aligned.csv",
    parse_dates=["pv_for_time", "load_for_time"]
)

# Build hourly actuals from 15-min data
raw = pd.read_csv(PROC / "all_houses_15min.csv", parse_dates=["timestamp"])
hourly = (raw.set_index("timestamp")
            .groupby("house_id")
            .resample("1h")
            .agg({"pv_kw":"mean","load_kw":"mean"})
            .reset_index())

# -------------------- PV EVALUATION (daylight only) -------------------
pv_eval = (pred[["house_id","pv_for_time","pv_nextday_kw_pred"]]
           .rename(columns={"pv_for_time":"target_time",
                            "pv_nextday_kw_pred":"pv_pred"}))

pv_eval = pv_eval.merge(
    hourly.rename(columns={"timestamp":"target_time"}),
    on=["house_id","target_time"], how="left"
)
pv_eval = pv_eval.rename(columns={"pv_kw":"pv_actual"})

# PV baseline: persistence (t-24h)
ref = hourly.copy()
ref["target_time"] = ref["timestamp"] + pd.Timedelta(hours=24)
pv_eval = pv_eval.merge(
    ref[["house_id","target_time","pv_kw"]]
        .rename(columns={"pv_kw":"pv_baseline"}),
    on=["house_id","target_time"], how="left"
)

# Mask out night / tiny actuals to avoid division blow-up
PV_MIN = 0.05  # kW
pv_mask = pv_eval["pv_actual"] >= PV_MIN

# -------------------- LOAD EVALUATION (suppress tiny) -----------------
ld_eval = (pred[["house_id","load_for_time","load_nextweek_kw_pred"]]
           .rename(columns={"load_for_time":"target_time",
                            "load_nextweek_kw_pred":"load_pred"}))

ld_eval = ld_eval.merge(
    hourly.rename(columns={"timestamp":"target_time"}),
    on=["house_id","target_time"], how="left"
)
ld_eval = ld_eval.rename(columns={"load_kw":"load_actual"})

# Load baseline: same weekday (t-168h)
refL = hourly.copy()
refL["target_time"] = refL["timestamp"] + pd.Timedelta(hours=168)
ld_eval = ld_eval.merge(
    refL[["house_id","target_time","load_kw"]]
        .rename(columns={"load_kw":"load_baseline"}),
    on=["house_id","target_time"], how="left"
)

LOAD_MIN = 0.10  # kW
ld_mask = ld_eval["load_actual"] >= LOAD_MIN

# ------------------------------ METRICS -------------------------------
def mae(y, yhat): return float(np.nanmean(np.abs(y - yhat)))

def mape_masked(y, yhat, min_actual):
    y = np.asarray(y); yhat = np.asarray(yhat)
    mask = np.abs(y) >= min_actual
    if not np.any(mask): return float("nan")
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / np.clip(np.abs(y[mask]), 1e-6, None))) * 100)

def smape(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    mask = denom > 1e-6
    if not np.any(mask): return float("nan")
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / denom[mask]) * 100)

summary = {
    # PV
    "PV_model_MAE_kW": mae(pv_eval.loc[pv_mask,"pv_actual"], pv_eval.loc[pv_mask,"pv_pred"]),
    "PV_base_MAE_kW":  mae(pv_eval.loc[pv_mask,"pv_actual"], pv_eval.loc[pv_mask,"pv_baseline"]),
    "PV_model_MAPE_%": mape_masked(pv_eval["pv_actual"], pv_eval["pv_pred"], PV_MIN),
    "PV_base_MAPE_%":  mape_masked(pv_eval["pv_actual"], pv_eval["pv_baseline"], PV_MIN),
    "PV_model_sMAPE_%": smape(pv_eval.loc[pv_mask,"pv_actual"], pv_eval.loc[pv_mask,"pv_pred"]),
    "PV_base_sMAPE_%":  smape(pv_eval.loc[pv_mask,"pv_actual"], pv_eval.loc[pv_mask,"pv_baseline"]),
    "PV_points_used":   int(pv_mask.sum()),
    # LOAD
    "Load_model_MAE_kW": mae(ld_eval.loc[ld_mask,"load_actual"], ld_eval.loc[ld_mask,"load_pred"]),
    "Load_base_MAE_kW":  mae(ld_eval.loc[ld_mask,"load_actual"], ld_eval.loc[ld_mask,"load_baseline"]),
    "Load_model_MAPE_%": mape_masked(ld_eval["load_actual"], ld_eval["load_pred"], LOAD_MIN),
    "Load_base_MAPE_%":  mape_masked(ld_eval["load_actual"], ld_eval["load_baseline"], LOAD_MIN),
    "Load_model_sMAPE_%": smape(ld_eval.loc[ld_mask,"load_actual"], ld_eval.loc[ld_mask,"load_pred"]),
    "Load_base_sMAPE_%":  smape(ld_eval.loc[ld_mask,"load_actual"], ld_eval.loc[ld_mask,"load_baseline"]),
    "Load_points_used":   int(ld_mask.sum())
}

# ------------------------- PER-HOUSE TABLES --------------------------
per_house = []
for hid, g in pv_eval[pv_mask].groupby("house_id"):
    per_house.append({
        "house_id": hid,
        "pv_model_mae": mae(g["pv_actual"], g["pv_pred"]),
        "pv_base_mae":  mae(g["pv_actual"], g["pv_baseline"]),
        "pv_model_sMAPE_%": smape(g["pv_actual"], g["pv_pred"]),
        "pv_base_sMAPE_%":  smape(g["pv_actual"], g["pv_baseline"]),
        "pv_points": len(g)
    })
for hid, g in ld_eval[ld_mask].groupby("house_id"):
    # update same record if exists
    row = next((r for r in per_house if r["house_id"]==hid), {"house_id":hid})
    row.update({
        "load_model_mae": mae(g["load_actual"], g["load_pred"]),
        "load_base_mae":  mae(g["load_actual"], g["load_baseline"]),
        "load_model_sMAPE_%": smape(g["load_actual"], g["load_pred"]),
        "load_base_sMAPE_%":  smape(g["load_actual"], g["load_baseline"]),
        "load_points": len(g)
    })
    if row not in per_house: per_house.append(row)

per_house_df = pd.DataFrame(per_house).sort_values("house_id")
per_house_df.to_csv(PROC / "agent1_validation_per_house.csv", index=False)

# ---------------------------- QUICK PLOTS -----------------------------
def plot_last_week(hid, kind="pv"):
    if kind == "pv":
        g = pv_eval[(pv_eval.house_id==hid) & pv_mask].copy()
        g = g.sort_values("target_time").tail(7*24)
        if g.empty: return
        plt.figure(figsize=(10,3))
        plt.plot(g["target_time"], g["pv_actual"], label="PV actual")
        plt.plot(g["target_time"], g["pv_pred"],   label="PV pred", linestyle="--")
        plt.title(f"House {hid} — PV day-ahead (last 7 days)"); plt.legend(); plt.tight_layout()
        plt.savefig(REPORTS / f"house_{hid:02d}_pv_last_week.png", dpi=130); plt.close()
    else:
        g = ld_eval[(ld_eval.house_id==hid) & ld_mask].copy()
        g = g.sort_values("target_time").tail(7*24)
        if g.empty: return
        plt.figure(figsize=(10,3))
        plt.plot(g["target_time"], g["load_actual"], label="Load actual")
        plt.plot(g["target_time"], g["load_pred"],   label="Load pred", linestyle="--")
        plt.title(f"House {hid} — Load week-ahead (last 7 days)"); plt.legend(); plt.tight_layout()
        plt.savefig(REPORTS / f"house_{hid:02d}_load_last_week.png", dpi=130); plt.close()

for hid in sorted(pred["house_id"].unique()):
    plot_last_week(hid, "pv")
    plot_last_week(hid, "load")

# -------------------------- SAVE SUMMARY -----------------------------
print(json.dumps(summary, indent=2))
(ROOT / "models" / "agent1_validation_summary.json").write_text(json.dumps(summary, indent=2))
print("Saved per-house CSV →", PROC/"agent1_validation_per_house.csv")
print("Saved plots →", REPORTS)
