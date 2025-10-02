import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd

RNG_SEED = 42


def load_interim(path: str) -> pd.DataFrame:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Interim parquet not found: {path}")
	return pd.read_parquet(path)


def pick_timestamp(df: pd.DataFrame, target_ts: str) -> pd.DataFrame:
	# ts_local should be parseable as datetime
	df = df.copy()
	df["ts_local"] = pd.to_datetime(df["ts_local"], utc=False)
	mask = df["ts_local"] == pd.Timestamp(target_ts)
	if not mask.any():
		raise ValueError(f"Timestamp {target_ts} not found in data")
	return df.loc[mask].reset_index(drop=True)


def build_buyers_sellers(
	df_ts: pd.DataFrame,
	grid_buy: float = 0.25,
	grid_sell: float = 0.08,
	buyer_bid_spread: float = 0.02,
	seller_ask_spread: float = 0.02,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	# Positive load -> buyer; positive pv -> seller
	buyers = []
	sellers = []
	for _, row in df_ts.iterrows():
		customer = str(row["Customer"]) if "Customer" in df_ts.columns else str(row.get("customer", ""))
		load_kwh = float(row.get("load_kwh", 0.0))
		pv_kwh = float(row.get("pv_kwh", 0.0))

		if load_kwh > 0.0:
			bid_price = max(grid_sell, grid_buy - buyer_bid_spread)
			buyers.append({"id": customer, "demand_kwh": load_kwh, "bid_price": bid_price})
		if pv_kwh > 0.0:
			ask_price = min(grid_buy, grid_sell + seller_ask_spread)
			sellers.append({"id": customer, "supply_kwh": pv_kwh, "ask_price": ask_price})

	buyers_df = pd.DataFrame(buyers)
	sellers_df = pd.DataFrame(sellers)
	return buyers_df, sellers_df


def build_losses(num_buyers: int, num_sellers: int) -> np.ndarray:
	rng = np.random.default_rng(RNG_SEED)
	# Deterministic, modest losses between 0% and 10%
	return rng.uniform(0.00, 0.10, size=(num_buyers, num_sellers)).astype(float)


def main():
	parser = argparse.ArgumentParser(description="Build buyers, sellers, and losses for one timestamp")
	parser.add_argument("--parquet", default=os.path.join("data", "interim", "ausgrid_30min.parquet"))
	parser.add_argument("--ts", required=True, help='Target timestamp, e.g., "2012-12-01 14:30"')
	parser.add_argument("--outdir", default=os.path.join("data", "interim"))
	args = parser.parse_args()

	df = load_interim(args.parquet)
	df_ts = pick_timestamp(df, args.ts)
	buyers_df, sellers_df = build_buyers_sellers(df_ts)

	losses = build_losses(len(buyers_df), len(sellers_df)) if len(buyers_df) and len(sellers_df) else np.zeros((len(buyers_df), len(sellers_df)))

	os.makedirs(args.outdir, exist_ok=True)
	buyers_path = os.path.join(args.outdir, "buyers.csv")
	sellers_path = os.path.join(args.outdir, "sellers.csv")
	losses_path = os.path.join(args.outdir, "losses.npy")

	buyers_df.to_csv(buyers_path, index=False)
	sellers_df.to_csv(sellers_path, index=False)
	np.save(losses_path, losses)

	print(f"Saved buyers -> {buyers_path} ({len(buyers_df)} rows)")
	print(f"Saved sellers -> {sellers_path} ({len(sellers_df)} rows)")
	print(f"Saved losses -> {losses_path} (shape={losses.shape})")


if __name__ == "__main__":
	main()
