"""Station 3 - funds and out-of-sample backtests for Better Finance.

The starter interface is kept unchanged.  The required combined equity-plus-crypto
fund is evaluated with equal-weight and minimum-variance methods, with risk parity
added as one modest extension.  All extra helpers are private.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _prepare_returns(returns: pd.DataFrame) -> pd.DataFrame:
    data = returns.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.set_index("date")

    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data.loc[~data.index.isna()].sort_index()
    data = data.loc[~data.index.duplicated(keep="first")]
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=1, how="all")

    if data.empty or data.shape[1] < 2:
        raise ValueError("returns must contain at least two usable asset series")
    return data


def _is_crypto(name: str) -> bool:
    text = str(name)
    return text.startswith("CR_") or text.endswith("-USD")


def _limit_crypto(weights: pd.Series, maximum: float = 0.25) -> pd.Series:
    result = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    if result.sum() <= 0:
        result[:] = 1.0
    result = result / result.sum()

    crypto_mask = pd.Series([_is_crypto(name) for name in result.index], index=result.index)
    crypto_total = float(result.loc[crypto_mask].sum())
    if not crypto_mask.any() or crypto_total <= maximum:
        return result

    equity_mask = ~crypto_mask
    result.loc[crypto_mask] = result.loc[crypto_mask] / crypto_total * maximum
    equity_total = float(result.loc[equity_mask].sum())
    if equity_total > 0:
        result.loc[equity_mask] = result.loc[equity_mask] / equity_total * (1.0 - maximum)
    return result / result.sum()


def _constraints(columns: pd.Index) -> list[dict]:
    rules = [{"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)}]
    crypto_mask = np.array([_is_crypto(name) for name in columns], dtype=bool)
    if crypto_mask.any() and (~crypto_mask).any():
        rules.append(
            {
                "type": "ineq",
                "fun": lambda x, mask=crypto_mask: float(0.25 - np.sum(x[mask])),
            }
        )
    return rules


def _minimum_variance(history: pd.DataFrame) -> pd.Series:
    sample = history.dropna(axis=1, thresh=max(20, int(len(history) * 0.90))).fillna(0.0)
    covariance = sample.cov().to_numpy(dtype=float)
    covariance = covariance + np.eye(covariance.shape[0]) * 1e-8
    number = sample.shape[1]
    start = _limit_crypto(pd.Series(1.0 / number, index=sample.columns)).to_numpy()
    bounds = [(0.0, 0.20)] * number

    answer = minimize(
        lambda x: float(x @ covariance @ x),
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=_constraints(sample.columns),
        options={"maxiter": 250, "ftol": 1e-9, "disp": False},
    )

    values = answer.x if answer.success and np.isfinite(answer.x).all() else start
    return _limit_crypto(pd.Series(values, index=sample.columns, dtype=float))


def _risk_parity(history: pd.DataFrame) -> pd.Series:
    sample = history.dropna(axis=1, thresh=max(20, int(len(history) * 0.90))).fillna(0.0)
    covariance = sample.cov().to_numpy(dtype=float)
    covariance = covariance + np.eye(covariance.shape[0]) * 1e-8
    number = sample.shape[1]
    start = _limit_crypto(pd.Series(1.0 / number, index=sample.columns)).to_numpy()
    bounds = [(0.0, 0.20)] * number

    def objective(x: np.ndarray) -> float:
        portfolio_variance = float(x @ covariance @ x)
        if portfolio_variance <= 1e-16:
            return 1e6
        contribution = x * (covariance @ x) / portfolio_variance
        target = np.repeat(1.0 / number, number)
        return float(np.square(contribution - target).sum())

    answer = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=_constraints(sample.columns),
        options={"maxiter": 300, "ftol": 1e-9, "disp": False},
    )

    values = answer.x if answer.success and np.isfinite(answer.x).all() else start
    return _limit_crypto(pd.Series(values, index=sample.columns, dtype=float))


def _equal_weight(columns: pd.Index) -> pd.Series:
    return _limit_crypto(pd.Series(1.0 / len(columns), index=columns, dtype=float))


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Calculate the fact-sheet measures required by the Project B brief."""
    values = pd.to_numeric(pd.Series(daily_returns), errors="coerce").dropna()
    if values.empty:
        return {
            "annualised_return": np.nan,
            "annualised_volatility": np.nan,
            "sharpe": np.nan,
            "maximum_drawdown": np.nan,
        }

    wealth = (1.0 + values).cumprod()
    years = len(values) / float(periods_per_year)
    annual_return = wealth.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    annual_volatility = values.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (
        values.mean() / values.std(ddof=1) * np.sqrt(periods_per_year)
        if values.std(ddof=1) > 0
        else np.nan
    )
    drawdown = wealth / wealth.cummax() - 1.0

    return {
        "annualised_return": float(annual_return),
        "annualised_volatility": float(annual_volatility),
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "maximum_drawdown": float(drawdown.min()),
    }


def oos_backtest(returns: pd.DataFrame, method: str = "min_variance"):
    """Run the monthly walk-forward OOS backtest required by the starter.

    A 252-observation training window is followed by 21-trading-day holding
    blocks.  The training window always ends before the rebalance date.  The
    portfolio is long-only, combined crypto exposure is capped at 25%, and
    transaction costs are assumed to be zero and reported explicitly.
    """
    data = _prepare_returns(returns)
    method = method.lower().strip()
    allowed = {"equal_weight", "min_variance", "risk_parity"}
    if method not in allowed:
        raise ValueError("method must be 'equal_weight', 'min_variance', or 'risk_parity'")
    if len(data) <= 253:
        raise ValueError("not enough observations for a 252-observation OOS backtest")

    daily_parts = []
    weight_rows = []
    audit_rows = []
    window = 252
    step = 21

    for start in range(window, len(data), step):
        history = data.iloc[start - window:start]
        future = data.iloc[start:min(start + step, len(data))]
        if future.empty:
            continue

        if method == "equal_weight":
            target = _equal_weight(history.columns)
        elif method == "min_variance":
            target = _minimum_variance(history)
        else:
            target = _risk_parity(history)

        live = future.reindex(columns=target.index).fillna(0.0)
        realised = live.mul(target, axis=1).sum(axis=1)
        daily_parts.append(realised)

        rebalance_date = pd.Timestamp(future.index[0])
        for ticker, weight in target.items():
            weight_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "holding_end": pd.Timestamp(future.index[-1]),
                    "ticker": str(ticker),
                    "weight": float(weight),
                }
            )

        audit_rows.append(
            {
                "rebalance_date": rebalance_date,
                "training_start": pd.Timestamp(history.index[0]),
                "training_end": pd.Timestamp(history.index[-1]),
                "holding_end": pd.Timestamp(future.index[-1]),
                "training_observations": int(len(history)),
                "holding_observations": int(len(future)),
                "past_only_check": bool(history.index[-1] < future.index[0]),
                "transaction_cost_bps": 0.0,
            }
        )

    if not daily_parts:
        raise ValueError("the OOS backtest produced no live holding periods")

    daily = pd.concat(daily_parts).sort_index()
    daily.name = "portfolio_return"
    return {
        "daily_returns": daily,
        "weights": pd.DataFrame(weight_rows),
        "audit": pd.DataFrame(audit_rows),
        "method": method,
    }
