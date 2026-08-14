"""Station 1 - your ETL: load and clean the data.

Load raw data through src.data_access (see context/DATA_GUIDE.md). Add your own
integrity checks. Do not commit data files.
"""
from src import data_access


def load_clean_equities():
    """Load equity prices and run your Station 1 integrity checks.

    The returned frame is unique by ``ticker`` and ``date`` and sorted in panel
    order. Genuine extreme returns are retained, as required by the brief. A
    compact audit is attached to ``df.attrs["integrity_audit"]`` so downstream
    code can report what was found without adding analysis-only columns to the
    clean price data.
    """
    import numpy as np
    import pandas as pd

    df = data_access.load_equity_prices().copy()
    rows_loaded = int(len(df))

    required = {
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjClose",
        "volume",
        "sector",
    }

    # Check whether numeric values are finite.
    numeric = df[
        ["open", "high", "low", "close", "adjClose", "volume"]
    ]

    non_finite_rows = int(
        (~np.isfinite(numeric)).any(axis=1).sum()
    )

    # Prices should normally be strictly positive.
    non_positive_price_rows = int(
        (
            df[
                ["open", "high", "low", "close", "adjClose"]
            ] <= 0
        )
        .any(axis=1)
        .sum()
    )

    # Trading volume should not be negative.
    negative_volume_rows = int(
        (df["volume"] < 0).sum()
    )

    # The daily high should not be below open, close, or low.
    # The daily low should not be above open, close, or high.
    inconsistent_ohlc_rows = int(
        (
            (
                df["high"]
                < df[["open", "close", "low"]].max(axis=1)
            )
            |
            (
                df["low"]
                > df[["open", "close", "high"]].min(axis=1)
            )
        ).sum()
    )

    missing_columns = sorted(required.difference(df.columns))

    if missing_columns:
        raise ValueError(
            f"equity data is missing required columns: {missing_columns}"
        )

    # Convert dates to a consistent timezone-naive datetime format.
    # News dates are timezone-aware, while price dates are normally not.
    df["date"] = (
        pd.to_datetime(df["date"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )

    invalid_dates = int(df["date"].isna().sum())
    missing_tickers = int(df["ticker"].isna().sum())
    missing_key_rows = int(
        (df["ticker"].isna() | df["date"].isna()).sum()
    )
    # Rows without a ticker or a valid date cannot be used in the price panel.
    df = df.dropna(subset=["ticker", "date"]).copy()

    # Equity prices should contain one row per ticker and trading date.
    duplicate_mask = df.duplicated(
        ["ticker", "date"],
        keep="first",
    )

    duplicate_rows = int(duplicate_mask.sum())

    df = (
        df.loc[~duplicate_mask]
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    # Build the shared equity trading calendar from all observed equity dates.
    trading_dates = pd.Index(
        df["date"]
        .drop_duplicates()
        .sort_values()
    )

    # Count observed trading dates for each ticker.
    observed_counts = (
        df.groupby("ticker", observed=True)["date"]
        .nunique()
    )

    # Count dates missing for each ticker relative to the shared equity calendar.
    missing_by_ticker = (
        len(trading_dates) - observed_counts
    ).astype(int)

    # Convert the missing-date counts into a DataFrame.
    missing_dates_table = (
        missing_by_ticker
        .rename("missing_date_count")
        .reset_index()
    )

    # Calculate daily stock returns only for integrity screening.
    stock_returns = (
        df.groupby("ticker", observed=True)["adjClose"]
        .pct_change(fill_method=None)
    )

    # Calculate ticker-specific lower and upper quantile thresholds.
    lower_bound = (
        stock_returns.groupby(df["ticker"], observed=True)
        .transform(lambda x: x.quantile(0.01))
    )

    upper_bound = (
        stock_returns.groupby(df["ticker"], observed=True)
        .transform(lambda x: x.quantile(0.99))
    )

    # Flag returns outside the ticker-specific 1st and 99th percentiles.
    extreme_return_mask = (
        stock_returns.lt(lower_bound)
        | stock_returns.gt(upper_bound)
    )

    extreme_return_rows = int(extreme_return_mask.sum())


    # Store the audit separately from the dataset columns.
    df.attrs["integrity_audit"] = {
        "dataset": "equity_prices",
        "source": "data_access.load_equity_prices",

        "rows_loaded": rows_loaded,
        "rows_clean": int(len(df)),
        "rows_removed": int(rows_loaded - len(df)),

        "missing_ticker_rows": missing_tickers,
        "invalid_date_rows": invalid_dates,
        "missing_key_rows_removed": missing_key_rows,
        "duplicate_rows_removed": duplicate_rows,

        "ticker_count": int(df["ticker"].nunique()),
        "sector_count": int(df["sector"].nunique()),
        "date_start": (
            df["date"].min().strftime("%Y-%m-%d")
            if not df.empty
            else None
        ),
        "date_end": (
            df["date"].max().strftime("%Y-%m-%d")
            if not df.empty
            else None
        ),
        "trading_date_count": int(len(trading_dates)),

        "tickers_with_missing_dates": int(
            missing_by_ticker.gt(0).sum()
        ),
        "missing_dates_total": int(
            missing_by_ticker.sum()
        ),
        "missing_dates_by_ticker": (
            missing_by_ticker.to_dict()
        ),

        "extreme_return_rows": extreme_return_rows,
        "extreme_return_rule": (
            "ticker-specific 1st/99th return percentiles"
        ),
        "extreme_return_action": "flagged and retained",

        "non_finite_numeric_rows": non_finite_rows,
        "non_positive_price_rows": non_positive_price_rows,
        "negative_volume_rows": negative_volume_rows,
        "inconsistent_ohlc_rows": inconsistent_ohlc_rows,

        "cleaning_actions": {
            "missing_keys": "removed",
            "duplicates": "keep first",
            "extreme_returns": "retained",
            "ohlcv_issues": "reported only",
        },
    }

    return df


def load_clean_crypto():
    """Load and clean crypto prices on their native seven-day calendar.

    Returns are intentionally not aligned to the equity calendar here. They must
    first be computed on the native crypto calendar, as specified in the brief.
    The observations dated after 2023-12-31 are removed, and the integrity
    findings are stored in ``df.attrs["integrity_audit"]``.
    """
    import numpy as np
    import pandas as pd

    df = data_access.load_crypto_prices().copy()
    rows_loaded = int(len(df))

    required = {
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjClose",
        "volume",
    }

    missing_columns = sorted(required.difference(df.columns))

    if missing_columns:
        raise ValueError(
            f"crypto data is missing required columns: {missing_columns}"
        )

    numeric = df[
        ["open", "high", "low", "close", "adjClose", "volume"]
    ]

    non_finite_rows = int(
        (~np.isfinite(numeric)).any(axis=1).sum()
    )

    non_positive_price_rows = int(
        (
            df[
                ["open", "high", "low", "close", "adjClose"]
            ] <= 0
        )
        .any(axis=1)
        .sum()
    )

    negative_volume_rows = int(
        (df["volume"] < 0).sum()
    )

    inconsistent_ohlc_rows = int(
        (
            (
                df["high"]
                < df[["open", "close", "low"]].max(axis=1)
            )
            |
            (
                df["low"]
                > df[["open", "close", "high"]].min(axis=1)
            )
        ).sum()
    )

    # Convert crypto dates into the same timezone-naive datetime format.
    df["date"] = (
        pd.to_datetime(df["date"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )

    invalid_dates = int(df["date"].isna().sum())
    missing_tickers = int(df["ticker"].isna().sum())

    # Count rows with either an unusable ticker or date.
    # Rows missing both fields are counted only once.
    missing_key_rows = int(
        (df["ticker"].isna() | df["date"].isna()).sum()
    )

    # The brief specifies that the project sample must end on 2023-12-31.
    after_sample = int(
        (df["date"] > pd.Timestamp("2023-12-31")).sum()
    )

    df = df.dropna(subset=["ticker", "date"]).copy()

    df = df.loc[
        df["date"] <= pd.Timestamp("2023-12-31")
    ].copy()

    # Crypto prices should also be unique by ticker and date.
    duplicate_mask = df.duplicated(
        ["ticker", "date"],
        keep="first",
    )

    duplicate_rows = int(duplicate_mask.sum())

    df = (
        df.loc[~duplicate_mask]
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    # Crypto trades every day, so use a complete daily calendar.
    full_calendar = pd.date_range(
        df["date"].min(),
        df["date"].max(),
        freq="D",
    )

    observed_counts = (
        df.groupby("ticker", observed=True)["date"]
        .nunique()
    )

    missing_by_ticker = (
        len(full_calendar) - observed_counts
    ).astype(int)

   # Calculate simple daily returns within each cryptocurrency.
    crypto_returns = (
        df.groupby(
            "ticker",
            observed=True,
        )["adjClose"]
        .pct_change(fill_method=None)
    )

    # Calculate ticker-specific lower and upper quantile thresholds.
    lower_return_bound = (
        crypto_returns
        .groupby(
            df["ticker"],
            observed=True,
        )
        .transform(
            lambda x: x.quantile(0.01)
        )
    )

    upper_return_bound = (
        crypto_returns
        .groupby(
            df["ticker"],
            observed=True,
        )
        .transform(
            lambda x: x.quantile(0.99)
        )
    )

    # Flag returns outside the ticker-specific 1st and 99th percentiles.
    # These observations are retained rather than removed.
    extreme_return_mask = (
        crypto_returns.lt(lower_return_bound)
        | crypto_returns.gt(upper_return_bound)
    )

    extreme_return_rows = int(
        extreme_return_mask.sum()
    )

    # Count how many cryptocurrencies contain at least one flagged return.
    tickers_with_extreme_returns = int(
        df.loc[
            extreme_return_mask,
            "ticker",
        ].nunique()
    )

    # Build a detailed reporting table.
    extreme_returns_table = (
        df.loc[
            extreme_return_mask,
            [
                "ticker",
                "date",
                "adjClose",
                "volume",
            ],
        ]
        .assign(
            return_value=crypto_returns.loc[
                extreme_return_mask
            ],
            lower_quantile=lower_return_bound.loc[
                extreme_return_mask
            ],
            upper_quantile=upper_return_bound.loc[
                extreme_return_mask
            ],
        )
        .reset_index(drop=True)
    )

    missing_dates_total = int(missing_by_ticker.sum())

    tickers_with_missing_dates = int(
        missing_by_ticker.gt(0).sum()
    )

    df.attrs["integrity_audit"] = {
        "dataset": "crypto_prices",

        # Row counts
        "rows_loaded": rows_loaded,
        "rows_clean": int(len(df)),
        "rows_removed": int(rows_loaded - len(df)),

        # Invalid key checks
        "invalid_dates": invalid_dates,
        "missing_tickers": missing_tickers,
        "missing_ticker_or_date_rows_removed": missing_key_rows,

        # Date-range cleaning
        "rows_after_2023_12_31_removed": after_sample,

        # Numeric and OHLCV integrity checks
        "non_finite_numeric_rows": non_finite_rows,
        "non_positive_price_rows": non_positive_price_rows,
        "negative_volume_rows": negative_volume_rows,
        "inconsistent_ohlc_rows": inconsistent_ohlc_rows,

        # Duplicate checks
        "duplicate_ticker_date_rows_removed": duplicate_rows,

        # Calendar checks
        "crypto_calendar_date_count": int(len(full_calendar)),
        "tickers_with_missing_dates": tickers_with_missing_dates,
        "missing_dates_total": missing_dates_total,
        "missing_dates_by_ticker": missing_by_ticker.to_dict(),

        # Extreme-return checks
        "extreme_return_rows_retained": extreme_return_rows,
        "tickers_with_extreme_returns": tickers_with_extreme_returns,
        "extreme_return_rule": (
            "Ticker-specific daily return below the 1st percentile "
            "or above the 99th percentile"
        ),
        "extreme_return_resolution": (
            "Flagged for review and retained in the dataset"
        ),
        "extreme_returns_table": (
            extreme_returns_table.to_dict(orient="records")
        ),

        # Dataset coverage
        "ticker_count": int(df["ticker"].nunique()),
        "date_start": (
            df["date"].min().strftime("%Y-%m-%d")
            if not df.empty
            else None
        ),
        "date_end": (
            df["date"].max().strftime("%Y-%m-%d")
            if not df.empty
            else None
        ),
    }

    return df

import pandas as pd

def load_clean_news() -> pd.DataFrame:
    """Load, clean and quality-check the news headline data."""

    news = data_access.load_news_headlines().copy()

    required = {
        "date",
        "ticker",
        "sector",
        "title",
        "url",
        "publisher",
    }

    missing_columns = sorted(
        required.difference(news.columns)
    )

    if missing_columns:
        raise ValueError(
            f"News data is missing columns: {missing_columns}"
        )

    # Record the number of rows before cleaning.
    raw_rows = len(news)

    # Clean string columns.
    for column in [
        "ticker",
        "sector",
        "title",
        "url",
        "publisher",
    ]:
        news[column] = (
            news[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    # Convert UTC-aware timestamps to timezone-naive daily dates.
    news["date"] = (
        pd.to_datetime(
            news["date"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # Count missing values after cleaning.
    missing_counts = (
        news[
            [
                "date",
                "ticker",
                "sector",
                "title",
                "url",
                "publisher",
            ]
        ]
        .isna()
        .sum()
        .astype(int)
        .to_dict()
    )

    # Rows missing these fields cannot be used later.
    unusable_rows = int(
        news[
            ["date", "ticker", "title"]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    news = news.dropna(
        subset=["date", "ticker", "title"]
    )

    # Only remove exact duplicate headlines.
    duplicate_key = [
        "ticker",
        "date",
        "title",
    ]

    duplicate_count = int(
        news.duplicated(
            subset=duplicate_key,
            keep="first",
        ).sum()
    )

    news = (
        news
        .drop_duplicates(
            subset=duplicate_key,
            keep="first",
        )
        .sort_values(
            ["date", "ticker", "title"]
        )
        .reset_index(drop=True)
    )

    # Store data-quality information in the DataFrame.
    news.attrs = {
        "raw_rows": raw_rows,
        "clean_rows": len(news),
        "rows_removed": raw_rows - len(news),
        "missing_counts": missing_counts,
        "unusable_rows_removed": unusable_rows,
        "duplicates_removed": duplicate_count,
        "date_start": news["date"].min(),
        "date_end": news["date"].max(),
        "frequency": (
            "Headline-level observations with daily dates"
        ),
        "schema": {
            column: str(dtype)
            for column, dtype in news.dtypes.items()
        },
        "provenance": (
            "Loaded through "
            "data_access.load_news_headlines(); "
            "text fields were stripped; "
            "UTC timestamps were converted to timezone-naive "
            "daily dates; exact duplicates were removed using "
            "ticker, date and title."
        ),
    }

    return news
