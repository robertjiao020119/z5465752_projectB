"""Station 2 - your features: return features and headline text assembly.

Build your return (and optional risk) features here, and assemble the headlines
into a daily text panel. Part A is assembly only - scoring the text into sentiment
(and lagging the signal) is the Station 3 model, built in Part B, not here.
"""
import pandas as pd
from src import data_access


def daily_returns(
    prices: pd.DataFrame,
    price_col: str = "adjClose",
) -> pd.DataFrame:
    """Compute simple daily returns within each ticker using adjusted close.

    The result stays in long form and preserves the original columns, with one
    added ``return`` column. Sorting before ``pct_change`` prevents returns from
    being calculated across tickers or out of chronological order. The first
    observation for each ticker remains missing because it has no previous price.
    """
    required = {
        "ticker",
        "date",
        price_col,
    }

    missing_columns = sorted(
        required.difference(prices.columns)
    )

    if missing_columns:
        raise ValueError(
            f"prices is missing required columns: {missing_columns}"
        )

    result = prices.copy()

    # Ensure dates have a consistent timezone-naive datetime type.
    result["date"] = (
        pd.to_datetime(result["date"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )

    if result["date"].isna().any():
        raise ValueError("prices contains invalid dates")

    # Duplicate ticker-date rows would make the return calculation ambiguous.
    if result.duplicated(["ticker", "date"]).any():
        raise ValueError(
            "prices must be unique by ticker and date "
            "before returns are computed"
        )

    result = (
        result.sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    #Transform to wide table and calculate

    wide_prices = (
        result
        .pivot(
            index="date",
            columns="ticker",
            values=price_col,
        )
        .sort_index()
        .sort_index(axis=1)
    )

    returns = (
        wide_prices
        .pct_change(fill_method=None)
        .dropna(how="all")
    )

    returns.index.name = "date"
    returns.columns.name = None

    # Identify crypto tickers by whether they contain weekend observations.
    weekend_by_ticker = (
        result.assign(
            is_weekend=result["date"].dt.dayofweek >= 5
        )
        .groupby("ticker")["is_weekend"]
        .any()
    )

    crypto_tickers = weekend_by_ticker[
        weekend_by_ticker
    ].index.tolist()

    equity_tickers = weekend_by_ticker[
        ~weekend_by_ticker
    ].index.tolist()

    def calculate_panel_returns(
        panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate returns on one asset panel's own calendar."""

        if panel.empty:
            return pd.DataFrame()

        panel_prices = (
            panel
            .pivot(
                index="date",
                columns="ticker",
                values=price_col,
            )
            .sort_index()
            .sort_index(axis=1)
        )

        return (
            panel_prices
            .pct_change(fill_method=None)
            .dropna(how="all")
        )

    # Calculate equity and crypto returns separately.
    equity_returns = calculate_panel_returns(
        result.loc[
            result["ticker"].isin(equity_tickers)
        ]
    )

    crypto_returns = calculate_panel_returns(
        result.loc[
            result["ticker"].isin(crypto_tickers)
        ]
    )

    # If both asset classes exist, align crypto returns only after
    # returns have been calculated on the full crypto calendar.
    if not equity_returns.empty and not crypto_returns.empty:
        returns = equity_returns.join(
            crypto_returns.reindex(equity_returns.index),
            how="left",
        )

    elif not equity_returns.empty:
        returns = equity_returns

    else:
        returns = crypto_returns


    return returns




def descriptive_stats_by_asset_class(
    returns: pd.DataFrame,
    equity_tickers: list[str],
    crypto_tickers: list[str],
) -> pd.DataFrame:
    """Summarize daily returns separately for equities and cryptocurrencies.

    Parameters
    ----------
    returns:
        Wide daily-return table with dates as the index and tickers as columns.
    equity_tickers:
        Tickers classified as equities.
    crypto_tickers:
        Tickers classified as cryptocurrencies.

    Returns
    -------
    pd.DataFrame
        One row per asset class containing mean, volatility, minimum,
        maximum, and skewness of daily returns.
    """
    asset_groups = {
        "Equity": equity_tickers,
        "Crypto": crypto_tickers,
    }

    rows = []

    for asset_class, tickers in asset_groups.items():
        valid_tickers = [
            ticker
            for ticker in tickers
            if ticker in returns.columns
        ]

        if not valid_tickers:
            raise ValueError(
                f"No valid tickers found for asset class: {asset_class}"
            )

        class_returns = (
            returns[valid_tickers]
            .stack()
            .dropna()
        )

        rows.append(
            {
                "asset_class": asset_class,
                "mean_return": class_returns.mean(),
                "volatility": class_returns.std(),
                "minimum": class_returns.min(),
                "maximum": class_returns.max(),
                "skewness": class_returns.skew(),
            }
        )

    return pd.DataFrame(rows)





def assemble_headline_panel(
    headlines: pd.DataFrame,
) -> pd.DataFrame:

    news = headlines.copy()
    required = {
        "date",
        "ticker",
        "sector",
        "title",
    }

    missing_columns = sorted(
        required.difference(headlines.columns)
    )

    if missing_columns:
        raise ValueError(
            f"headlines is missing required columns: {missing_columns}"
        )

    # Load equity dates to obtain the actual trading calendar.
    equity_prices = data_access.load_equity_prices()

    trading_dates = pd.DatetimeIndex(
        equity_prices["date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    positions = trading_dates.searchsorted(
        news["date"],
        side="left",
    )

    # Remove headlines after the last available equity trading date.
    valid = positions < len(trading_dates)

    result = news.iloc[valid].copy()

    # Same trading day when available;
    # otherwise, the next equity trading day.
    result["date"] = trading_dates[
        positions[valid]
    ].to_numpy()

    panel = (
        result
        .groupby(
            ["date", "ticker", "sector"],
            as_index=False,
        )
        .agg(
            headline_count=("title", "size"),
            text=("title", "\n".join),
        )
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    return panel
