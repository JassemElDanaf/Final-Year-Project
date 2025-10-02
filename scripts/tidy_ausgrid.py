import argparse
import os

import pandas as pd

RAW_RELATIVE = os.path.join("data", "raw", "Solar_2012_2013", "2012-2013 Solar home electricity data v2.csv")
OUT_PARQUET = os.path.join("data", "interim", "ausgrid_30min.parquet")


HALF_HOUR_COLS = [
	"0:30","1:00","1:30","2:00","2:30","3:00","3:30","4:00","4:30","5:00","5:30","6:00","6:30",
	"7:00","7:30","8:00","8:30","9:00","9:30","10:00","10:30","11:00","11:30","12:00","12:30","13:00",
	"13:30","14:00","14:30","15:00","15:30","16:00","16:30","17:00","17:30","18:00","18:30","19:00",
	"19:30","20:00","20:30","21:00","21:30","22:00","22:30","23:00","23:30","0:00"
]


def tidy(raw_csv: str, out_parquet: str) -> None:
	# Skip first metadata row
	df = pd.read_csv(raw_csv, skiprows=1)
	# Melt half-hour columns to long
	id_vars = ["Customer", "Generator Capacity", "Postcode", "Consumption Category", "date"]
	df_long = df.melt(id_vars=id_vars, value_vars=HALF_HOUR_COLS, var_name="hh", value_name="kwh")

	# Build timestamp
	# 'date' is in d/m/Y; 'hh' is like '14:30' or '0:00'
	# Compose ts_local as pandas datetime without timezone
	# Handle possible trailing commas by coercion
	df_long["date"] = pd.to_datetime(df_long["date"], dayfirst=True)
	# Normalize '0:00' representing midnight end of day -> same date 00:00
	df_long["hh"] = df_long["hh"].astype(str)
	df_long["ts_local"] = pd.to_datetime(df_long["date"].dt.strftime("%Y-%m-%d") + " " + df_long["hh"], errors="coerce")

	# Split by category: GC = consumption, GG = generation, CL = controlled load
	df_long = df_long.rename(columns={"Consumption Category": "cons_cat"})

	cons = df_long.loc[df_long["cons_cat"] == "GC", ["Customer", "ts_local", "kwh"]].rename(columns={"kwh": "load_kwh"})
	gen = df_long.loc[df_long["cons_cat"] == "GG", ["Customer", "ts_local", "kwh"]].rename(columns={"kwh": "pv_kwh"})

	# Merge consumption and generation
	merged = pd.merge(cons, gen, on=["Customer", "ts_local"], how="outer")
	merged["load_kwh"] = merged["load_kwh"].fillna(0.0).astype(float)
	merged["pv_kwh"] = merged["pv_kwh"].fillna(0.0).astype(float)

	os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
	merged.to_parquet(out_parquet, index=False)
	print(f"Wrote {out_parquet} with {len(merged):,} rows")


def main():
	parser = argparse.ArgumentParser(description="Tidy Ausgrid raw CSV to 30-min parquet")
	parser.add_argument("--raw", default=RAW_RELATIVE)
	parser.add_argument("--out", default=OUT_PARQUET)
	args = parser.parse_args()

	tidy(args.raw, args.out)


if __name__ == "__main__":
	main()
