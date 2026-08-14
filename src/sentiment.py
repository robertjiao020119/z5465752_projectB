"""Station 3 - plain VADER sentiment index for Better Finance.

This implementation stays close to the course baseline: preserve headline text,
score with VADER, average to ticker-day, then equal-weight tickers inside each
sector.  No-news ticker-days are neutral and the tradable series is lagged one
trading date.
"""
from __future__ import annotations

import re

import pandas as pd


def _find_columns(panel: pd.DataFrame) -> tuple[str, str]:
    date_column = "date" if "date" in panel.columns else "trading_date"
    if "text" in panel.columns:
        text_column = "text"
    elif "headline_text" in panel.columns:
        text_column = "headline_text"
    elif "title" in panel.columns:
        text_column = "title"
    else:
        raise ValueError("headline panel needs text, headline_text, or title")
    return date_column, text_column


def _vader():
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def _titles(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"\n+|\s*\|\|\s*", str(text)) if item.strip()]


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply plain VADER and return one average sentiment score per ticker-day."""
    date_column, text_column = _find_columns(panel)
    analyzer = _vader()
    rows = []

    for row in panel[[date_column, "ticker", "sector", text_column]].itertuples(index=False, name=None):
        day, ticker, sector, text = row
        titles = _titles(text)
        values = [analyzer.polarity_scores(title)["compound"] for title in titles]
        if values:
            rows.append(
                {
                    "date": pd.Timestamp(day),
                    "ticker": str(ticker),
                    "sector": str(sector),
                    "sentiment": float(sum(values) / len(values)),
                    "headline_count": int(len(values)),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("headline panel contains no usable text")
    result = result.sort_values(["date", "sector", "ticker"]).reset_index(drop=True)
    if "trading_calendar" in panel.attrs:
        result.attrs["trading_calendar"] = pd.DatetimeIndex(
            pd.to_datetime(panel.attrs["trading_calendar"], errors="coerce")
        ).dropna().sort_values().unique()
    return result


def sector_sentiment_index(scores: pd.DataFrame) -> pd.DataFrame:
    """Build the equal-ticker-weight sector index with neutral no-news days."""
    required = {"date", "ticker", "sector", "sentiment", "headline_count"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"scores is missing required columns: {missing}")

    data = scores.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "ticker", "sector"])

    ticker_map = data[["ticker", "sector"]].drop_duplicates()
    source_calendar = scores.attrs.get("trading_calendar")
    if source_calendar is None:
        calendar_values = sorted(data["date"].unique())
    else:
        calendar_values = list(pd.DatetimeIndex(source_calendar).sort_values())
    dates = pd.DataFrame({"date": calendar_values})
    dates["_key"] = 1
    ticker_map = ticker_map.copy()
    ticker_map["_key"] = 1
    grid = dates.merge(ticker_map, on="_key").drop(columns="_key")
    grid = grid.merge(data, on=["date", "ticker", "sector"], how="left")

    grid["has_news"] = grid["sentiment"].notna().astype(int)
    grid["sentiment"] = grid["sentiment"].fillna(0.0)
    grid["headline_count"] = grid["headline_count"].fillna(0).astype(int)

    sector = (
        grid.groupby(["date", "sector"], as_index=False)
        .agg(
            sentiment=("sentiment", "mean"),
            coverage=("has_news", "mean"),
            headline_count=("headline_count", "sum"),
        )
        .sort_values(["sector", "date"])
        .reset_index(drop=True)
    )
    sector["lagged_sentiment"] = sector.groupby("sector")["sentiment"].shift(1).fillna(0.0)
    sector["sentiment_0_100"] = (sector["sentiment"] + 1.0) * 50.0
    return sector
