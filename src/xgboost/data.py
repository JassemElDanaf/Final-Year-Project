from pathlib import Path
import pandas as pd, numpy as np, re
from .config import RAW, PROC

def _find_one(folder: Path, patterns):
    for f in folder.iterdir():
        if f.is_file():
            name = f.name.lower()
            for pat in patterns:
                if re.search(pat, name, flags=re.IGNORECASE):
                    return f
    raise FileNotFoundError(f"No matching file in {folder} for {patterns}")

def load_load(hpath: Path):
    f = _find_one(hpath, [r'^load house \d+\.csv$', r'^load\.csv$'])
    df = pd.read_csv(f)
    dt_col = next((c for c in df.columns if c.lower() in ("datetime","timestamp")
                  or ("date" in c.lower() and "time" in c.lower())), None)
    if dt_col is None and {"Date","Time"}.issubset(df.columns):
        df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"]); dt_col = "DateTime"
    ts = pd.to_datetime(df[dt_col])
    ycol = next((c for c in df.columns if "consumption" in c.lower()
                or ("load" in c.lower() and "kw" in c.lower())), None)
    y = pd.to_numeric(df[ycol], errors="coerce").replace(-1, np.nan)
    return pd.DataFrame({"timestamp": ts, "load_kw": y})

def load_pv(hpath: Path):
    f = _find_one(hpath, [r'^pv generation house \d+\.csv$', r'^pv.*house \d+\.csv$', r'^pv\.csv$'])
    df = pd.read_csv(f)
    tcol = next(c for c in df.columns if "time" in c.lower() or "date" in c.lower())
    pcol = next(c for c in df.columns if "pv" in c.lower() or "(w)" in c.lower())
    return pd.DataFrame({"timestamp": pd.to_datetime(df[tcol]),
                         "pv_kw": pd.to_numeric(df[pcol], errors="coerce")/1000})

def load_weather(hpath: Path):
    f = _find_one(hpath, [r'^weather house \d+\.csv$', r'^weather\.csv$'])
    df = pd.read_csv(f)
    tcol = next(c for c in df.columns if "time" in c.lower() or "date" in c.lower())
    def pick(*opts):
        for o in opts:
            m = [c for c in df.columns if o.lower() in c.lower()]
            if m: return pd.to_numeric(df[m[0]], errors="coerce")
        return np.nan
    return pd.DataFrame({
        "timestamp": pd.to_datetime(df[tcol]),
        "ghi": pick("GHI"),
        "temp_c": pick("temp","temperature"),
        "rh": pick("humidity"),
        "wind_ms": pick("wind speed"),
        "cloud_pct": pick("cloud cover","clouds"),
    })

def load_price():
    f = _find_one(RAW / "market", [r'^price portugal.*\.csv$', r'^price_.*\.csv$'])
    df = pd.read_csv(f)
    tcol = next(c for c in df.columns if "time" in c.lower() or "date" in c.lower())
    pcol = next(c for c in df.columns if "price" in c.lower())
    return (pd.DataFrame({"timestamp": pd.to_datetime(df[tcol]),
                          "price_eur_kwh": pd.to_numeric(df[pcol], errors="coerce")/1000})
            .sort_values("timestamp"))

def build_processed_for_all(save_per_house=True):
    price = load_price()
    houses = sorted([p for p in RAW.glob("house_*") if p.is_dir()])
    all_list = []
    for p in houses:
        hid = int(p.name.split("_")[1])
        L, PV, W = load_load(p), load_pv(p), load_weather(p)
        df = (L.merge(PV, on="timestamp", how="outer")
                .merge(W,  on="timestamp", how="outer")
                .merge(price, on="timestamp", how="left"))
        # use new pandas freq alias '15min'
        df = (df.sort_values("timestamp")
                .set_index("timestamp")
                .resample("15min").mean())
        df["house_id"] = hid
        df["dow"]   = df.index.dayofweek
        df["hour"]  = df.index.hour
        df["month"] = df.index.month
        df["surplus_kw"] = df["pv_kw"].fillna(0) - df["load_kw"].fillna(0)
        out = df.reset_index()
        if save_per_house:
            out.to_csv(PROC / f"house_{hid:02d}_15min.csv", index=False)
        all_list.append(out)
    all_df = pd.concat(all_list, ignore_index=True)
    all_df.to_csv(PROC / "all_houses_15min.csv", index=False)
    return all_df
