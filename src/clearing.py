import math
from typing import Optional

import numpy as np
import pandas as pd


def clear_cheapest_first(
    buyers_df: pd.DataFrame,
    sellers_df: pd.DataFrame,
    losses: Optional[np.ndarray] = None,
    grid_buy: float = 0.25,
    grid_sell: float = 0.08,
) -> pd.DataFrame:
    """
    Greedy clearing that matches buyers to sellers by ascending effective delivered price.

    buyers_df: columns [id, demand_kwh, bid_price]
    sellers_df: columns [id, supply_kwh, ask_price]
    losses: buyers x sellers loss fraction in [0,1] representing delivery loss
    grid_buy: price to buy from grid (fallback for buyers)
    grid_sell: price to sell to grid (fallback for sellers)

    Returns allocations with columns: [buyer_id, seller_id, kwh, eff_price]
      - kwh is delivered energy to buyer
      - eff_price is price per delivered kWh paid by buyer
    """
    required_buyer_cols = {"id", "demand_kwh", "bid_price"}
    required_seller_cols = {"id", "supply_kwh", "ask_price"}

    if not required_buyer_cols.issubset(buyers_df.columns):
        raise ValueError(f"buyers_df must have columns {required_buyer_cols}")
    if not required_seller_cols.issubset(sellers_df.columns):
        raise ValueError(f"sellers_df must have columns {required_seller_cols}")

    buyers = buyers_df.copy().reset_index(drop=True)
    sellers = sellers_df.copy().reset_index(drop=True)

    num_buyers = len(buyers)
    num_sellers = len(sellers)

    if losses is None:
        losses = np.zeros((num_buyers, num_sellers), dtype=float)
    else:
        losses = np.asarray(losses, dtype=float)
        if losses.shape != (num_buyers, num_sellers):
            raise ValueError(
                f"losses must have shape ({num_buyers}, {num_sellers}); got {losses.shape}"
            )
        losses = np.clip(losses, 0.0, 0.95)  # cap extreme losses

    remaining_demand = buyers["demand_kwh"].astype(float).to_numpy()
    remaining_supply = sellers["supply_kwh"].astype(float).to_numpy()

    buyer_bid = buyers["bid_price"].astype(float).to_numpy()
    seller_ask = sellers["ask_price"].astype(float).to_numpy()

    # Build candidate list (buyer_index, seller_index, effective_delivered_price)
    candidates = []
    for bi in range(num_buyers):
        for si in range(num_sellers):
            if remaining_demand[bi] <= 0 and buyers.loc[bi, "demand_kwh"] > 0:
                continue
            if remaining_supply[si] <= 0 and sellers.loc[si, "supply_kwh"] > 0:
                continue
            loss_frac = float(losses[bi, si])
            deliver_efficiency = max(1e-9, 1.0 - loss_frac)

            # Price bounded by buyer bid and grid price from buyer perspective, not below seller ask
            base_price = max(seller_ask[si], min(buyer_bid[bi], grid_buy))
            # Effective price per delivered kWh accounts for having to buy more to deliver 1 kWh
            eff_price = base_price / deliver_efficiency
            candidates.append((bi, si, eff_price))

    # Sort by cheapest effective delivered price
    candidates.sort(key=lambda x: x[2])

    allocations: list[tuple[str, str, float, float]] = []

    for bi, si, eff_price in candidates:
        if remaining_demand[bi] <= 0 or remaining_supply[si] <= 0:
            continue
        loss_frac = float(losses[bi, si])
        deliver_efficiency = max(1e-9, 1.0 - loss_frac)

        # Max deliverable energy from this seller given supply and losses
        max_deliverable_from_seller = remaining_supply[si] * deliver_efficiency
        if max_deliverable_from_seller <= 0:
            continue

        delivered = min(remaining_demand[bi], max_deliverable_from_seller)
        if delivered <= 0:
            continue

        # Energy drawn at seller side to deliver 'delivered'
        draw_from_seller = delivered / deliver_efficiency

        remaining_demand[bi] -= delivered
        remaining_supply[si] -= draw_from_seller

        allocations.append(
            (str(buyers.loc[bi, "id"]), str(sellers.loc[si, "id"]), float(delivered), float(eff_price))
        )

    # Fallback to grid for any unmet demand
    for bi in range(num_buyers):
        if remaining_demand[bi] > 1e-9:
            delivered = float(remaining_demand[bi])
            allocations.append((str(buyers.loc[bi, "id"]), "GRID", delivered, float(grid_buy)))
            remaining_demand[bi] = 0.0

    # Sell excess to grid for any remaining supply
    for si in range(num_sellers):
        if remaining_supply[si] > 1e-9:
            delivered = float(remaining_supply[si])  # no loss modeled to grid
            # Represent as buyer GRID purchasing from seller
            allocations.append(("GRID", str(sellers.loc[si, "id"]), delivered, float(grid_sell)))
            remaining_supply[si] = 0.0

    alloc_df = pd.DataFrame(allocations, columns=["buyer_id", "seller_id", "kwh", "eff_price"]).copy()
    return alloc_df
