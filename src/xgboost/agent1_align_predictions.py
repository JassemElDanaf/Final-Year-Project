import pandas as pd
from pathlib import Path

PROC = Path(r"E:/FYP/data/processed")
df = pd.read_csv(PROC/"agent1_predictions.csv", parse_dates=["timestamp"])

out = df.copy()
out["pv_for_time"]   = out["timestamp"] + pd.Timedelta(hours=24)
out["load_for_time"] = out["timestamp"] + pd.Timedelta(hours=24*7)

# physical clipping
out["pv_nextday_kw_pred"]    = out["pv_nextday_kw_pred"].clip(lower=0)
out["load_nextweek_kw_pred"] = out["load_nextweek_kw_pred"].clip(lower=0)

# optional: keep only the aligned columns you’ll use downstream
out = out[[
    "pv_for_time","house_id","pv_nextday_kw_pred",
    "load_for_time","load_nextweek_kw_pred"
]]

out.to_csv(PROC/"agent1_predictions_aligned.csv", index=False)
print("Saved →", PROC/"agent1_predictions_aligned.csv")
