"""Better Finance - simple investor dashboard for Project B.

The deployed app reads precomputed files from results/.  It does not run VADER,
SciPy optimisation, or the walk-forward backtest at interaction time.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

from src import data_access  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "results" / "data"
TABLES = ROOT / "results" / "tables"

st.set_page_config(page_title="Better Finance", page_icon="💵", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background:linear-gradient(180deg,#f7f9fd 0%,#eef2f8 100%);}
    .bf-hero {
        padding:1.25rem 1.4rem;
        border-radius:18px;
        background:linear-gradient(120deg,#24355f 0%,#445b9c 58%,#7181bf 100%);
        border:1px solid #314779;
        box-shadow:0 10px 24px rgba(38,54,96,.14);
        margin-bottom:1rem;
    }
    .bf-hero h1 {margin:0;color:#ffffff;font-size:2.05rem;letter-spacing:.01em;}
    .bf-hero p {margin:.38rem 0 0;color:#e7ebf7;}
    .small-note {color:#66728a;font-size:.86rem;}
    div[data-testid="stMetric"] {
        background:#ffffff;
        border:1px solid #dce3ef;
        padding:.78rem;
        border-radius:14px;
        box-shadow:0 4px 14px rgba(35,51,88,.06);
    }
    div[data-testid="stMetricLabel"] {color:#56637c;}
    div[data-testid="stMetricValue"] {color:#23345f;}
    div[data-baseweb="tab-list"] {gap:.35rem;}
    button[data-baseweb="tab"] {
        background:#f1f4fa;
        border-radius:10px 10px 0 0;
        padding:.55rem .85rem;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background:#e1e7f5;
        color:#283d73;
        font-weight:600;
    }
    div.stButton > button {
        border:1px solid #536aa8;
        color:#304a86;
        border-radius:10px;
    }
    div.stButton > button:hover {
        border-color:#304a86;
        color:#23345f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=86_400, show_spinner="Loading data...")
def _equities():
    """Starter-compatible hosted equity loader used only in the Data page."""
    return data_access.load_equity_prices()


@st.cache_data
def _load_outputs():
    returns = pd.read_csv(DATA / "fund_returns.csv", parse_dates=["date"])
    weights = pd.read_csv(DATA / "fund_weights.csv", parse_dates=["rebalance_date"])
    sector = pd.read_csv(DATA / "sector_sentiment_index.csv", parse_dates=["date"])
    metrics = pd.read_csv(TABLES / "performance_metrics.csv")
    comparison_path = TABLES / "fusion_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    return returns, weights, sector, metrics, comparison


def _percentage(value) -> str:
    return "N/A" if pd.isna(value) else f"{float(value) * 100:.2f}%"


def _latest_holdings(weights: pd.DataFrame, fund: str) -> pd.DataFrame:
    rows = weights.loc[weights["fund"].eq(fund)].copy()
    if rows.empty:
        return rows
    last_date = rows["rebalance_date"].max()
    return rows.loc[rows["rebalance_date"].eq(last_date)].sort_values("weight", ascending=False)


def _risk_label(volatility: float, all_volatility: pd.Series) -> str:
    if pd.isna(volatility):
        return "Not available"
    lower = all_volatility.quantile(0.33)
    upper = all_volatility.quantile(0.67)
    if volatility <= lower:
        return "Lower relative risk"
    if volatility <= upper:
        return "Medium relative risk"
    return "Higher relative risk"


try:
    fund_returns, fund_weights, sector_sentiment, performance, fusion_table = _load_outputs()
except FileNotFoundError as exc:
    st.error(f"Project B output is missing: {exc}. Run python scripts/run_part_b.py first.")
    st.stop()

st.markdown(
    """
    <div class="bf-hero">
      <h1>Better Finance</h1>
      <p>A simple way to compare systematic equity-and-crypto funds, check risk, and follow the mood of financial news.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

compare_tab, fact_tab, allocation_tab, sentiment_tab, about_tab = st.tabs(
    ["Compare Funds", "Fund Details", "My Allocation", "News Mood", "About & Data"]
)

with compare_tab:
    st.subheader("Compare Better Finance funds")
    st.caption("All performance shown below is walk-forward out-of-sample performance. Risk-free rate is assumed to be zero.")

    best_sharpe = performance.sort_values("sharpe", ascending=False).iloc[0]
    low_risk = performance.sort_values("annualised_volatility").iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Highest Sharpe", best_sharpe["fund"], f"{best_sharpe['sharpe']:.2f}")
    c2.metric("Lowest volatility", low_risk["fund"], _percentage(low_risk["annualised_volatility"]))
    c3.metric("Funds available", str(len(performance)), "combined equity + crypto")

    display = performance[
        ["fund", "annualised_return", "annualised_volatility", "sharpe", "maximum_drawdown"]
    ].copy()
    display["risk_level"] = display["annualised_volatility"].apply(
        lambda x: _risk_label(x, performance["annualised_volatility"])
    )
    display["annualised_return"] *= 100
    display["annualised_volatility"] *= 100
    display["maximum_drawdown"] *= 100
    display = display.rename(
        columns={
            "fund": "Fund",
            "annualised_return": "Annual return (%)",
            "annualised_volatility": "Volatility (%)",
            "sharpe": "Sharpe",
            "maximum_drawdown": "Max drawdown (%)",
            "risk_level": "Simple risk guide",
        }
    )
    st.dataframe(display.round(2), hide_index=True, width="stretch")

    selected = st.multiselect(
        "Choose funds for the growth comparison",
        performance["fund"].tolist(),
        default=performance["fund"].tolist(),
    )
    if selected:
        growth = (1.0 + fund_returns.set_index("date")[selected].fillna(0.0)).cumprod()
        st.line_chart(growth, height=430)

    st.markdown("#### Quick guide")
    goal = st.selectbox(
        "What matters most to you?",
        ["Lower volatility", "Higher historical risk-adjusted return", "See the news-sentiment version"],
    )
    if goal == "Lower volatility":
        choice = performance.sort_values("annualised_volatility").iloc[0]
    elif goal == "Higher historical risk-adjusted return":
        choice = performance.sort_values("sharpe", ascending=False).iloc[0]
    else:
        news_rows = performance.loc[performance["fund"].str.contains("Sentiment", na=False)]
        choice = news_rows.iloc[0] if not news_rows.empty else performance.iloc[0]
    st.info(
        f"A fund to inspect based on your selected preference: **{choice['fund']}**."
    )

with fact_tab:
    st.subheader("Fund fact sheet")
    fund = st.selectbox("Choose a fund", performance["fund"].tolist(), key="fact_fund")
    row = performance.loc[performance["fund"].eq(fund)].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualised return", _percentage(row["annualised_return"]))
    c2.metric("Annualised volatility", _percentage(row["annualised_volatility"]))
    c3.metric("Sharpe ratio", f"{row['sharpe']:.2f}")
    c4.metric("Maximum drawdown", _percentage(row["maximum_drawdown"]))
    st.caption(_risk_label(row["annualised_volatility"], performance["annualised_volatility"]))

    series = fund_returns.set_index("date")[fund].dropna()
    wealth = (1.0 + series).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    left, right = st.columns(2)
    with left:
        st.write("**Growth of $1**")
        st.line_chart(wealth, height=360)
    with right:
        st.write("**Drawdown**")
        st.area_chart(drawdown, height=360)

    st.write("**Current target holdings**")
    holdings = _latest_holdings(fund_weights, fund)
    if holdings.empty:
        st.info("No holdings file is available for this fund.")
    else:
        holdings = holdings[["ticker", "asset_class", "sector", "weight"]].copy()
        holdings["weight"] *= 100
        st.dataframe(
            holdings.rename(columns={"weight": "Weight (%)"}).round(2),
            hide_index=True,
            width="stretch",
            height=390,
        )

with allocation_tab:
    st.subheader("Build a simple allocation")
    st.caption("Enter a hypothetical amount for each fund. The chart combines their historical OOS return series.")
    fund_list = performance["fund"].tolist()
    amounts = []
    columns = st.columns(2)
    for i, fund in enumerate(fund_list):
        with columns[i % 2]:
            amounts.append(
                st.number_input(
                    fund,
                    min_value=0.0,
                    value=1000.0 if i < 2 else 0.0,
                    step=100.0,
                    key=f"alloc_{i}",
                )
            )

    total = float(sum(amounts))
    if total > 0:
        shares = np.asarray(amounts, dtype=float) / total
        allocation = pd.DataFrame(
            {
                "Fund": fund_list,
                "Amount ($)": amounts,
                "Allocation (%)": shares * 100,
            }
        )
        st.dataframe(allocation.round(2), hide_index=True, width="stretch")

        aligned = fund_returns.set_index("date")[fund_list].fillna(0.0)
        blended = aligned.mul(shares, axis=1).sum(axis=1)
        blend_growth = (1.0 + blended).cumprod()
        blend_vol = blended.std(ddof=1) * np.sqrt(252)
        blend_sharpe = blended.mean() / blended.std(ddof=1) * np.sqrt(252) if blended.std(ddof=1) > 0 else np.nan
        c1, c2, c3 = st.columns(3)
        c1.metric("Total amount", f"${total:,.0f}")
        c2.metric("Historical OOS volatility", _percentage(blend_vol))
        c3.metric("Historical OOS Sharpe", "N/A" if pd.isna(blend_sharpe) else f"{blend_sharpe:.2f}")
        st.line_chart(blend_growth, height=400)
    else:
        st.info("Enter a positive amount for at least one fund.")

with sentiment_tab:
    st.subheader("Equity sector news mood")
    st.caption(
        "Better Finance uses plain VADER on headlines. Ticker-days without headlines are treated as neutral, and the fund tilt uses the previous trading day's sector sentiment."
    )

    latest_date = sector_sentiment["date"].max()
    latest = sector_sentiment.loc[sector_sentiment["date"].eq(latest_date), ["sector", "sentiment", "coverage"]].copy()
    latest = latest.set_index("sector").sort_values("sentiment")
    st.write(f"**Latest sector reading: {latest_date:%Y-%m-%d}**")
    st.bar_chart(latest[["sentiment"]], height=330)

    sector = st.selectbox("Choose a sector", sorted(sector_sentiment["sector"].unique()))
    block = sector_sentiment.loc[sector_sentiment["sector"].eq(sector)].set_index("date")
    st.line_chart(block[["sentiment", "lagged_sentiment"]], height=390)
    st.write("Headline coverage")
    st.line_chart(block[["coverage"]], height=220)

    if not fusion_table.empty:
        st.write("**Minimum Variance before vs after sentiment**")
        show = fusion_table.copy()
        st.dataframe(show.round(4), hide_index=True, width="stretch")
        delta = float(show.iloc[0]["sharpe_delta"])
        if delta >= 0:
            st.success(f"The simple sentiment tilt improved OOS Sharpe by {delta:.3f} in this sample.")
        else:
            st.warning(f"The simple sentiment tilt reduced OOS Sharpe by {abs(delta):.3f}. The negative result is kept rather than hidden.")

with about_tab:
    st.subheader("About Better Finance")
    st.write(
        "Better Finance continues the Project A idea of making financial information easier for people who have limited time or technical knowledge. "
        "Project B turns the cleaned equity, crypto and headline data into investable combined funds, simple fact sheets, an allocation view and a sector news index."
    )
    st.markdown(
        """
        **Backtest design**
        - 252-observation rolling training window.
        - Rebalance every 21 equity trading days.
        - Long-only portfolio; combined crypto exposure capped at 25%.
        - Risk-free rate assumed to be zero for Sharpe.
        - Transaction cost assumed to be zero and stated explicitly.
        - Sentiment is equity-only and lagged one trading day before use.
        """
    )

    st.write("**Hosted data check**")
    if st.button("Load hosted equity summary"):
        try:
            eq = _equities()
            st.success(
                f"Equity prices loaded: {len(eq):,} rows, {eq['ticker'].nunique()} tickers, "
                f"{pd.to_datetime(eq['date']).min():%Y-%m-%d} to {pd.to_datetime(eq['date']).max():%Y-%m-%d}."
            )
        except Exception as exc:
            st.info(f"Hosted data is temporarily unavailable: {exc}")

