"""Station 3 - simple structured/unstructured fusion for Better Finance."""
from __future__ import annotations

import numpy as np
import pandas as pd


_TILT = 0.15


def _latest_signals(sentiment: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    available = sentiment.loc[pd.to_datetime(sentiment["date"]) <= pd.Timestamp(date)]
    if available.empty:
        return pd.Series(dtype=float)
    return (
        available.sort_values("date")
        .groupby("sector", as_index=False)
        .tail(1)
        .set_index("sector")["lagged_sentiment"]
    )


def apply_sentiment(weights: pd.DataFrame, sentiment: pd.DataFrame):
    """Tilt equity weights toward sectors with better lagged headline sentiment.

    The rule is intentionally simple and transparent.  Crypto receives no news
    score, so its total sleeve is held fixed.  Equity names are multiplied by
    ``1 + 0.15 * lagged_sector_sentiment`` and then re-normalised.
    """
    result_blocks = []
    data = weights.copy()
    data["rebalance_date"] = pd.to_datetime(data["rebalance_date"])

    for date, block in data.groupby("rebalance_date"):
        part = block.copy()
        signal = _latest_signals(sentiment, date)

        if "sector" not in part.columns:
            part["sector"] = "Unknown"
        if "asset_class" not in part.columns:
            part["asset_class"] = np.where(
                part["ticker"].astype(str).str.startswith("CR_")
                | part["ticker"].astype(str).str.endswith("-USD"),
                "Crypto",
                "Equity",
            )

        part["base_weight"] = part["weight"].astype(float)
        part["signal_used"] = part["sector"].map(signal).fillna(0.0)
        equity = part["asset_class"].eq("Equity")

        part.loc[equity, "weight"] = (
            part.loc[equity, "weight"]
            * (1.0 + _TILT * part.loc[equity, "signal_used"])
        ).clip(lower=0.0)

        crypto_total = float(part.loc[~equity, "base_weight"].sum())
        equity_budget = 1.0 - crypto_total
        if equity.any() and part.loc[equity, "weight"].sum() > 0:
            part.loc[equity, "weight"] = (
                part.loc[equity, "weight"]
                / part.loc[equity, "weight"].sum()
                * equity_budget
            )
        if (~equity).any() and crypto_total > 0:
            part.loc[~equity, "weight"] = part.loc[~equity, "base_weight"]

        part["weight"] = part["weight"] / part["weight"].sum()
        result_blocks.append(part)

    if not result_blocks:
        return data.copy()
    return pd.concat(result_blocks, ignore_index=True).sort_values(["rebalance_date", "ticker"])
