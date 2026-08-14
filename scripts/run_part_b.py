"""Reproduce Better Finance Part B results. Run from the project root:

    python scripts/run_part_b.py
"""
from __future__ import annotations

import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import data_access, etl, features, fusion, portfolios, sentiment  # noqa: E402


_PRODUCT_NAME = "Better Finance"
_METHODS = [
    ("equal_weight", "Equal Weight"),
    ("min_variance", "Minimum Variance"),
    ("risk_parity", "Risk Parity"),
]


def _read_returns(path: pathlib.Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_column = "date" if "date" in frame.columns else frame.columns[0]
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=[date_column]).set_index(date_column).sort_index()
    return frame.apply(pd.to_numeric, errors="coerce")


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse this student's Project A outputs where possible.

    Project A saved complete return matrices but only a small headline sample, so
    the full headline panel is rebuilt through this student's unchanged Project A
    ETL/features functions when the Part B model is reproduced locally.
    """
    candidates: list[pathlib.Path] = []

    override = os.environ.get("PROJECT_A_RESULTS_DIR")
    if override:
        candidates.append(pathlib.Path(override).expanduser())

    suffix = "_projectB"
    if PROJECT_ROOT.name.endswith(suffix):
        zid = PROJECT_ROOT.name[: -len(suffix)]
        candidates.append(PROJECT_ROOT.parent / f"{zid}_projectA" / "results" / "data")

    candidates.append(PROJECT_ROOT / "part_a_results")

    part_a_data = next((folder for folder in candidates if folder.exists()), None)
    stock_file = part_a_data / "stock_returns.csv" if part_a_data is not None else None
    crypto_file = part_a_data / "crypto_returns.csv" if part_a_data is not None else None

    if stock_file is not None and crypto_file is not None and stock_file.exists() and crypto_file.exists():
        stock_returns = _read_returns(stock_file)
        crypto_returns = _read_returns(crypto_file)
        source = f"reused Project A return artifacts from {part_a_data}"
    else:
        equities = etl.load_clean_equities()
        crypto = etl.load_clean_crypto()
        stock_returns = features.daily_returns(equities)
        crypto_returns = features.daily_returns(crypto)
        source = "rebuilt with unchanged Project A etl/features"

    combined = pd.concat(
        [
            stock_returns.add_prefix("EQ_"),
            crypto_returns.reindex(stock_returns.index).add_prefix("CR_"),
        ],
        axis=1,
    )
    combined.index.name = "date"

    if hasattr(etl, "load_clean_news"):
        news = etl.load_clean_news()
    else:
        news = data_access.load_news_headlines()
    headline_panel = features.assemble_headline_panel(news)
    headline_panel.attrs["foundation_source"] = source
    headline_panel.attrs["trading_calendar"] = pd.DatetimeIndex(stock_returns.index)
    return combined, headline_panel


def _sector_map(headline_panel: pd.DataFrame) -> dict[str, str]:
    return (
        headline_panel[["ticker", "sector"]]
        .drop_duplicates()
        .set_index("ticker")["sector"]
        .astype(str)
        .to_dict()
    )


def _decorate(weights: pd.DataFrame, fund_name: str, method_name: str, sector_map: dict[str, str]) -> pd.DataFrame:
    answer = weights.copy()
    answer["fund"] = fund_name
    answer["asset_family"] = "Combined"
    answer["method"] = method_name

    clean_ticker = answer["ticker"].astype(str).str.replace(r"^(EQ_|CR_)", "", regex=True)
    crypto = answer["ticker"].astype(str).str.startswith("CR_") | clean_ticker.str.endswith("-USD")
    answer["asset_class"] = np.where(crypto, "Crypto", "Equity")
    answer["sector"] = np.where(
        crypto,
        "Crypto",
        clean_ticker.map(sector_map).fillna("Unknown"),
    )
    return answer


def _returns_from_weights(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    dates = sorted(pd.to_datetime(weights["rebalance_date"].unique()))
    result = []

    for position, date in enumerate(dates):
        next_date = dates[position + 1] if position + 1 < len(dates) else None
        block = weights.loc[pd.to_datetime(weights["rebalance_date"]) == date]
        target = block.set_index("ticker")["weight"].astype(float)
        future = returns.loc[returns.index >= date]
        if next_date is not None:
            future = future.loc[future.index < next_date]
        if future.empty:
            continue
        realised = (
            future.reindex(columns=target.index)
            .fillna(0.0)
            .mul(target, axis=1)
            .sum(axis=1)
        )
        result.append(realised)

    if not result:
        return pd.Series(dtype=float, name="portfolio_return")
    series = pd.concat(result).sort_index()
    series.name = "portfolio_return"
    return series


def _save_figure(fig: plt.Figure, path: pathlib.Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_figures(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    metrics: pd.DataFrame,
    sector_index: pd.DataFrame,
    comparison: pd.DataFrame,
    folder: pathlib.Path,
) -> None:
    wealth = (1.0 + returns.set_index("date")).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    wealth.plot(ax=ax, linewidth=1.2)
    ax.set_title("Better Finance — OOS Growth of $1")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    _save_figure(fig, folder / "growth_of_one_dollar.png")

    chosen = "Combined Minimum Variance"
    if chosen in wealth.columns:
        drawdown = wealth[chosen] / wealth[chosen].cummax() - 1.0
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.25)
        ax.plot(drawdown.index, drawdown.values, linewidth=1.0)
        ax.set_title("Better Finance — Minimum Variance Drawdown")
        ax.set_ylabel("Drawdown")
        ax.set_xlabel("Date")
        _save_figure(fig, folder / "drawdown.png")

    crypto_weights = (
        weights.assign(
            crypto_weight=np.where(weights["asset_class"].eq("Crypto"), weights["weight"], 0.0)
        )
        .groupby(["rebalance_date", "method"], as_index=False)["crypto_weight"]
        .sum()
        .pivot(index="rebalance_date", columns="method", values="crypto_weight")
    )
    if not crypto_weights.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        crypto_weights.plot(ax=ax)
        ax.set_title("Combined Funds — Crypto Weight Over Time")
        ax.set_ylabel("Weight")
        ax.set_xlabel("Rebalance date")
        _save_figure(fig, folder / "portfolio_weights_over_time.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = metrics.sort_values("sharpe")
    ax.barh(order["fund"], order["sharpe"])
    ax.set_title("Better Finance — Sharpe Ratio by Fund")
    ax.set_xlabel("Sharpe ratio (rf = 0)")
    _save_figure(fig, folder / "sharpe_by_fund_method.png")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, block in sector_index.groupby("sector"):
        ax.plot(block["date"], block["sentiment"], label=name, linewidth=0.85)
    ax.set_title("Better Finance — Equity Sector Sentiment")
    ax.set_ylabel("VADER compound score")
    ax.set_xlabel("Date")
    ax.legend(fontsize=6, ncol=2)
    _save_figure(fig, folder / "sector_sentiment_index.png")

    if not comparison.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        values = [comparison.iloc[0]["base_sharpe"], comparison.iloc[0]["fused_sharpe"]]
        ax.bar(["Base Min Variance", "With Sentiment"], values)
        ax.set_title("Sentiment Fusion — Before vs After")
        ax.set_ylabel("Sharpe ratio")
        _save_figure(fig, folder / "fusion_before_after.png")


def main() -> None:
    data_folder = PROJECT_ROOT / "results" / "data"
    table_folder = PROJECT_ROOT / "results" / "tables"
    figure_folder = PROJECT_ROOT / "results" / "figures"
    for folder in [data_folder, table_folder, figure_folder]:
        folder.mkdir(parents=True, exist_ok=True)

    combined_returns, headline_panel = _inputs()
    sector_map = _sector_map(headline_panel)

    fund_series: dict[str, pd.Series] = {}
    weight_frames = []
    metric_rows = []
    audit_frames = []

    for method, method_name in _METHODS:
        fund_name = f"Combined {method_name}"
        print("Running", fund_name)
        outcome = portfolios.oos_backtest(combined_returns, method=method)
        series = outcome["daily_returns"]
        fund_series[fund_name] = series

        decorated = _decorate(outcome["weights"], fund_name, method_name, sector_map)
        weight_frames.append(decorated)

        measures = portfolios.performance_metrics(series, periods_per_year=252)
        measures.update(
            {
                "fund": fund_name,
                "asset_family": "Combined",
                "method": method_name,
                "first_live_date": series.dropna().index.min(),
                "last_live_date": series.dropna().index.max(),
                "transaction_cost_bps": 0.0,
            }
        )
        metric_rows.append(measures)

        audit = outcome["audit"].copy()
        audit["fund"] = fund_name
        audit_frames.append(audit)

    ticker_scores = sentiment.score_headlines(headline_panel)
    sector_index = sentiment.sector_sentiment_index(ticker_scores)

    all_weights = pd.concat(weight_frames, ignore_index=True)
    base_name = "Combined Minimum Variance"
    base_weights = all_weights.loc[all_weights["fund"].eq(base_name)].copy()
    sentiment_weights = fusion.apply_sentiment(base_weights, sector_index)
    sentiment_name = "Combined Minimum Variance + Sentiment"
    sentiment_weights["fund"] = sentiment_name
    sentiment_weights["method"] = "Minimum Variance + Sentiment"
    sentiment_weights["asset_family"] = "Combined"

    fused_returns = _returns_from_weights(combined_returns, sentiment_weights)
    fund_series[sentiment_name] = fused_returns
    weight_frames.append(sentiment_weights)

    fused_metrics = portfolios.performance_metrics(fused_returns, periods_per_year=252)
    fused_metrics.update(
        {
            "fund": sentiment_name,
            "asset_family": "Combined",
            "method": "Minimum Variance + Sentiment",
            "first_live_date": fused_returns.dropna().index.min(),
            "last_live_date": fused_returns.dropna().index.max(),
            "transaction_cost_bps": 0.0,
        }
    )
    metric_rows.append(fused_metrics)

    performance = pd.DataFrame(metric_rows)
    all_weights = pd.concat(weight_frames, ignore_index=True)
    fund_returns = (
        pd.concat(fund_series, axis=1)
        .sort_index()
        .reset_index()
        .rename(columns={"index": "date"})
    )

    base_metrics = performance.loc[performance["fund"].eq(base_name)].iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "base_fund": base_name,
                "fused_fund": sentiment_name,
                "base_annualised_return": base_metrics["annualised_return"],
                "fused_annualised_return": fused_metrics["annualised_return"],
                "return_delta": fused_metrics["annualised_return"] - base_metrics["annualised_return"],
                "base_sharpe": base_metrics["sharpe"],
                "fused_sharpe": fused_metrics["sharpe"],
                "sharpe_delta": fused_metrics["sharpe"] - base_metrics["sharpe"],
                "base_max_drawdown": base_metrics["maximum_drawdown"],
                "fused_max_drawdown": fused_metrics["maximum_drawdown"],
            }
        ]
    )

    fund_returns.to_csv(data_folder / "fund_returns.csv", index=False)
    all_weights.to_csv(data_folder / "fund_weights.csv", index=False)
    sector_index.to_csv(data_folder / "sector_sentiment_index.csv", index=False)
    ticker_scores.to_csv(data_folder / "ticker_day_sentiment.csv", index=False)
    performance.to_csv(table_folder / "performance_metrics.csv", index=False)
    comparison.to_csv(table_folder / "fusion_comparison.csv", index=False)
    pd.concat(audit_frames, ignore_index=True).to_csv(table_folder / "rebalance_audit.csv", index=False)

    metadata = performance[["fund", "asset_family", "method", "first_live_date", "last_live_date"]].copy()
    metadata["product"] = _PRODUCT_NAME
    metadata["target_user"] = "Busy investors who want simple, understandable fund comparisons"
    metadata["transaction_cost_bps"] = 0.0
    metadata.to_csv(data_folder / "fund_metadata.csv", index=False)

    _save_figures(
        fund_returns,
        all_weights,
        performance,
        sector_index,
        comparison,
        figure_folder,
    )
    print("Better Finance Project B outputs completed.")
    print("Required files written under results/data and results/tables.")


if __name__ == "__main__":
    main()
