import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd

from src.clearing import clear_cheapest_first
from scripts.build_buyers_sellers import load_interim, pick_timestamp, build_buyers_sellers, build_losses


def compute_kpis(
	alloc_df: pd.DataFrame,
	buyers_df: pd.DataFrame,
	sellers_df: pd.DataFrame,
) -> Dict[str, float]:
	# Costs for buyers: sum(kwh * eff_price) where buyer_id != 'GRID'
	alloc = alloc_df.copy()
	buyer_spend = (alloc.loc[alloc["buyer_id"] != "GRID", "kwh"] * alloc.loc[alloc["buyer_id"] != "GRID", "eff_price"]).sum()
	grid_kwh_to_buyers = alloc.loc[alloc["buyer_id"] != "GRID"].groupby("buyer_id")["kwh"].sum().sum()

	# Grid deliveries to buyers
	grid_to_buyers_kwh = alloc.loc[alloc["seller_id"] == "GRID", "kwh"].sum()

	# Seller revenue: include sales to buyers and grid
	seller_rev = alloc.loc[alloc["seller_id"] != "GRID", ["seller_id", "kwh", "eff_price"]]
	seller_rev_amt = (seller_rev["kwh"] * seller_rev["eff_price"]).sum()
	grid_sales_from_sellers_kwh = alloc.loc[alloc["buyer_id"] == "GRID", "kwh"].sum()

	# Unserved demand should be zero due to grid fallback
	total_demand = buyers_df["demand_kwh"].sum()
	delivered_to_buyers = alloc.loc[alloc["buyer_id"] != "GRID", "kwh"].sum() + grid_to_buyers_kwh
	unserved = max(0.0, float(total_demand - delivered_to_buyers))

	return {
		"system_total_cost": float(buyer_spend),
		"avg_buyer_bill": float(buyer_spend / max(1e-9, len(buyers_df))),
		"seller_revenue": float(seller_rev_amt),
		"grid_kwh": float(grid_to_buyers_kwh + grid_sales_from_sellers_kwh),
		"unserved_kwh": float(unserved),
	}


def main():
	parser = argparse.ArgumentParser(description="Run cheapest-first baseline")
	parser.add_argument("--parquet", default=os.path.join("data", "interim", "ausgrid_30min.parquet"))
	parser.add_argument("--ts", default="2012-12-01 14:30")
	parser.add_argument("--outdir", default=os.path.join("data", "processed"))
	args = parser.parse_args()

	df = load_interim(args.parquet)
	df_ts = pick_timestamp(df, args.ts)
	buyers_df, sellers_df = build_buyers_sellers(df_ts)
	losses = build_losses(len(buyers_df), len(sellers_df)) if len(buyers_df) and len(sellers_df) else np.zeros((len(buyers_df), len(sellers_df)))

	alloc_df = clear_cheapest_first(buyers_df, sellers_df, losses=losses)

	os.makedirs(args.outdir, exist_ok=True)
	outfile = os.path.join(args.outdir, f"alloc_{pd.Timestamp(args.ts).strftime('%Y%m%d_%H%M')}.csv")
	alloc_df.to_csv(outfile, index=False)

	kpis = compute_kpis(alloc_df, buyers_df, sellers_df)
	print("KPIs:")
	for k, v in kpis.items():
		print(f"  {k}: {v:.4f}")
	print(f"Saved allocations -> {outfile} ({len(alloc_df)} rows)")


if __name__ == "__main__":
	main()
