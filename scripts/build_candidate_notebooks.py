"""Build the flat `_candidates/` notebook catalog.

The candidate catalog is intentionally flat: every workflow notebook lives
directly under `_candidates/<number>_<slug>.ipynb`. Existing executed core and
buy-side notebooks are copied in, and new candidate workflows are generated as
self-contained notebooks with real QuantJourney SDK calls plus local pandas /
numpy analytics.
"""

from __future__ import annotations

import ast
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "_candidates"
ADVANCED = ROOT / "notebooks" / "buy_side_advanced"
MANIFEST = ROOT / "outputs" / "manifest.json"


EXISTING_CORE = [
    "01_authentication_methods.ipynb",
    "02_market_data_basics.ipynb",
    "03_economic_data_macro.ipynb",
    "04_fundamental_analysis.ipynb",
    "05_technical_analysis.ipynb",
    "06_portfolio_analysis.ipynb",
    "07_crypto_ccxt.ipynb",
    "08_cboe_vix.ipynb",
    "09_multpl_valuation.ipynb",
    "10_cftc_cot.ipynb",
]


EXISTING_BUY_SIDE = [
    "20_multi_factor_model.ipynb",
    "21_volatility_surface_greeks.ipynb",
    "23_cot_positioning_sentiment.ipynb",
    "24_macro_regime_allocation.ipynb",
    "25_cross_asset_correlation.ipynb",
    "26_risk_parity_portfolio.ipynb",
    "27_var_expected_shortfall.ipynb",
    "28_factor_attribution.ipynb",
    "29_pairs_trading_stat_arb.ipynb",
    "31_event_study_earnings.ipynb",
    "35_performance_reporting.ipynb",
    "40_sector_rotation_momentum.ipynb",
    "43_tail_risk_hedging.ipynb",
    "51_factor_risk_attribution.ipynb",
    "59_risk_adjusted_performance.ipynb",
]


COMMON_SETUP = r'''
import os
import math
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quantjourney.sdk import QuantJourneyAPI

qj = QuantJourneyAPI.from_env()

START = os.getenv("QJ_EXAMPLE_START", "2020-01-01")
END = os.getenv("QJ_EXAMPLE_END") or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

plt.style.use("default")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def unwrap(payload: Any) -> Any:
    """Return the useful data value from common QuantJourney response shapes."""
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if isinstance(payload, dict) and "value" in payload:
        return payload["value"]
    return payload


def safe_call(label: str, fn, **kwargs) -> Any:
    """Run an SDK call and keep the notebook usable when an optional feed is unavailable."""
    try:
        out = fn(**kwargs)
        print(f"{label}: ok")
        return out
    except Exception as exc:
        print(f"{label}: unavailable ({type(exc).__name__}: {exc})")
        return None


def as_rows(payload: Any) -> list[dict[str, Any]]:
    value = unwrap(payload)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("rows", "data", "items", "prices", "results"):
            if isinstance(value.get(key), list):
                return value[key]
        return [value]
    return []


def price_frame(symbol: str, start: str = START, end: str = END) -> pd.DataFrame:
    payload = qj.eod.get_historical_prices(symbol=symbol, start_date=start, end_date=end)
    rows = as_rows(payload)
    if not rows and isinstance(unwrap(payload), dict):
        value = unwrap(payload)
        rows = value.get(symbol) or value.get(symbol.upper()) or []
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No price data returned for {symbol}")
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "adjusted_close" in df and df["adjusted_close"].notna().any():
        df["price"] = df["adjusted_close"].fillna(df["close"])
    else:
        df["price"] = df["close"]
    if "volume" not in df:
        df["volume"] = np.nan
    return df.dropna(subset=["price"]).sort_values("date").set_index("date")


def price_panel(symbols: list[str], start: str = START, end: str = END) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = {}
    volumes = {}
    for symbol in symbols:
        df = price_frame(symbol, start=start, end=end)
        prices[symbol] = df["price"]
        volumes[symbol] = df["volume"]
    return pd.DataFrame(prices).dropna(how="all"), pd.DataFrame(volumes).reindex(pd.DataFrame(prices).index)


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")


def max_drawdown(nav: pd.Series) -> float:
    drawdown = nav / nav.cummax() - 1
    return float(drawdown.min())


def performance_stats(ret: pd.Series) -> pd.Series:
    ret = ret.dropna()
    nav = (1 + ret).cumprod()
    ann_ret = nav.iloc[-1] ** (252 / len(ret)) - 1 if len(ret) and nav.iloc[-1] > 0 else np.nan
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol and np.isfinite(ann_vol) else np.nan
    return pd.Series({
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(nav) if len(nav) else np.nan,
        "total_return": nav.iloc[-1] - 1 if len(nav) else np.nan,
    })


def inverse_vol_weights(ret: pd.DataFrame) -> pd.Series:
    vol = ret.std().replace(0, np.nan)
    inv = 1 / vol
    return (inv / inv.sum()).fillna(0)


def min_variance_weights(ret: pd.DataFrame, ridge: float = 1e-4) -> pd.Series:
    cov = ret.cov().fillna(0).to_numpy() * 252
    cov = cov + np.eye(cov.shape[0]) * ridge
    inv = np.linalg.pinv(cov)
    raw = inv @ np.ones(cov.shape[0])
    raw = np.maximum(raw, 0)
    if raw.sum() == 0:
        raw = np.ones(cov.shape[0])
    return pd.Series(raw / raw.sum(), index=ret.columns)


def portfolio_returns(ret: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned = ret[weights.index].dropna()
    return aligned @ weights.reindex(aligned.columns).fillna(0)


def risk_contribution(ret: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned = ret[weights.index].dropna()
    cov = aligned.cov() * 252
    w = weights.reindex(cov.columns).fillna(0).to_numpy()
    port_var = float(w @ cov.to_numpy() @ w)
    if port_var <= 0:
        return pd.Series(0.0, index=cov.columns)
    contrib = w * (cov.to_numpy() @ w) / port_var
    return pd.Series(contrib, index=cov.columns)


def rolling_betas(y: pd.Series, x: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    data = pd.concat([y.rename("asset"), x], axis=1).dropna()
    rows = []
    for i in range(window, len(data)):
        chunk = data.iloc[i - window:i]
        yy = chunk["asset"].to_numpy()
        xx = np.column_stack([np.ones(len(chunk)), chunk[x.columns].to_numpy()])
        beta = np.linalg.lstsq(xx, yy, rcond=None)[0][1:]
        rows.append(dict(date=data.index[i], **{col: beta[j] for j, col in enumerate(x.columns)}))
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=x.columns)


def zscore(s: pd.Series, window: int = 252) -> pd.Series:
    return (s - s.rolling(window).mean()) / s.rolling(window).std()


def dollar_adv(prices: pd.DataFrame, volumes: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    return (prices * volumes).rolling(window).mean()


def plot_nav(ret_map: dict[str, pd.Series], title: str) -> None:
    fig, ax = plt.subplots()
    for label, ret in ret_map.items():
        nav = (1 + ret.dropna()).cumprod()
        ax.plot(nav.index, nav, label=label)
    ax.set_title(title)
    ax.legend()
    plt.show()
'''


@dataclass(frozen=True)
class NotebookSpec:
    filename: str
    title: str
    category: str
    summary: str
    data_calls: list[str]
    cells: list[str]
    mirror_to_advanced: bool = False


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    source = dedent(text).strip() + "\n"
    ast.parse(source)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def notebook(spec: NotebookSpec) -> dict:
    intro = f"""
    # {spec.title}

    {spec.summary}

    **Category:** {spec.category}

    **Primary API calls used in this candidate:**
    {chr(10).join(f"- `{call}`" for call in spec.data_calls)}

    **Prepared by:** QuantJourney

    The notebook is self-contained: QuantJourney SDK calls fetch the data, while the research logic is calculated in pandas/numpy so the assumptions are visible and auditable. Candidate notebooks use direct connector SDK methods when a workflow needs provider-specific data; production systems can expose the same workflows through governed domain routes, tenant scopes and audit metadata.
    """
    cells = [md(intro), code(COMMON_SETUP)]
    cells.extend(code(cell) for cell in spec.cells)
    cells.append(md("""
    ## Notes

    This is a candidate workflow. In production, tenant scopes, connector allowlists,
    provider metadata, request IDs and audit logs should be retained next to the resulting
    tables or charts.
    """))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SPECS = [
    NotebookSpec(
        "22_hierarchical_risk_parity_hrp.ipynb",
        "Hierarchical Risk Parity vs Risk Parity vs Minimum Variance",
        "Portfolio construction",
        "Compares HRP-style recursive allocation, inverse-volatility risk parity and minimum variance using the same price panel.",
        ["qj.eod.get_historical_prices"],
        [
            """
            symbols = ["SPY", "TLT", "GLD", "DBC", "UUP"]
            prices, volumes = price_panel(symbols)
            ret = returns(prices).dropna()
            ret.tail()
            """,
            """
            def corr_sort_order(ret: pd.DataFrame) -> list[str]:
                corr = ret.corr().fillna(0)
                score = corr.mean().sort_values()
                return list(score.index)

            def cluster_var(cov: pd.DataFrame, assets: list[str]) -> float:
                sub = cov.loc[assets, assets]
                w = inverse_vol_weights(ret[assets])
                return float(w @ sub @ w)

            def hrp_weights(ret: pd.DataFrame) -> pd.Series:
                cov = ret.cov() * 252
                order = corr_sort_order(ret)
                weights = pd.Series(1.0, index=order)
                clusters = [order]
                while clusters:
                    cluster = clusters.pop(0)
                    if len(cluster) <= 1:
                        continue
                    split = len(cluster) // 2
                    left, right = cluster[:split], cluster[split:]
                    left_var = cluster_var(cov, left)
                    right_var = cluster_var(cov, right)
                    alpha = 1 - left_var / (left_var + right_var)
                    weights[left] *= alpha
                    weights[right] *= 1 - alpha
                    clusters.extend([left, right])
                return weights.reindex(ret.columns).fillna(0) / weights.sum()

            weights = pd.DataFrame({
                "HRP": hrp_weights(ret),
                "InverseVol": inverse_vol_weights(ret),
                "MinVariance": min_variance_weights(ret),
            })
            weights
            """,
            """
            portfolio = {col: portfolio_returns(ret, weights[col]) for col in weights.columns}
            summary = pd.DataFrame({name: performance_stats(series) for name, series in portfolio.items()}).T
            display(summary)
            display(weights.style.format("{:.2%}"))
            plot_nav(portfolio, "HRP vs inverse-vol vs min-variance")
            risk = pd.DataFrame({col: risk_contribution(ret, weights[col]) for col in weights.columns})
            risk.plot(kind="bar", title="Risk contribution by method")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "30_universe_construction_liquidity_screen.ipynb",
        "Universe Construction and Liquidity Screen",
        "Universe construction",
        "Builds an investable equity universe from prices, volumes, optional FMP screener output, TTM ratios and short-interest context.",
        ["qj.eod.get_historical_prices", "qj.fmp.get_stock_screener", "qj.fmp.get_financial_ratios_ttm", "qj.finra.get_short_interest"],
        [
            """
            seed_symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "JPM", "LLY", "XOM", "UNH", "V"]
            screener = safe_call("FMP screener", qj.fmp.get_stock_screener, marketCapMoreThan=10_000_000_000, limit=50)
            prices, volumes = price_panel(seed_symbols)
            ret = returns(prices)
            adv = dollar_adv(prices, volumes).iloc[-1]
            """,
            """
            ratio_rows = []
            for symbol in seed_symbols:
                payload = safe_call(f"FMP ratios {symbol}", qj.fmp.get_financial_ratios_ttm, symbol=symbol)
                data = unwrap(payload) or {}
                ratio_rows.append({
                    "symbol": symbol,
                    "pe_ttm": pd.to_numeric(data.get("peRatioTTM"), errors="coerce"),
                    "gross_margin_ttm": pd.to_numeric(data.get("grossProfitMarginTTM"), errors="coerce"),
                    "fcf_yield_ttm": 1 / pd.to_numeric(data.get("priceToFreeCashFlowRatioTTM"), errors="coerce")
                    if data.get("priceToFreeCashFlowRatioTTM") else np.nan,
                })
            ratios = pd.DataFrame(ratio_rows).set_index("symbol")
            short_interest = {
                symbol: safe_call(f"FINRA short interest {symbol}", qj.finra.get_short_interest, symbol=symbol)
                for symbol in seed_symbols[:5]
            }
            """,
            """
            features = pd.DataFrame(index=seed_symbols)
            features["adv_usd"] = adv.reindex(seed_symbols)
            features["volatility_63d"] = ret[seed_symbols].tail(63).std() * np.sqrt(252)
            features["momentum_126d"] = prices[seed_symbols].pct_change(126).iloc[-1]
            features = features.join(ratios)
            features["liquid"] = features["adv_usd"] > 50_000_000
            features["quality_score"] = features["gross_margin_ttm"].rank(pct=True) + features["fcf_yield_ttm"].rank(pct=True)
            features["risk_penalty"] = features["volatility_63d"].rank(pct=True)
            features["universe_score"] = features["quality_score"] + features["momentum_126d"].rank(pct=True) - features["risk_penalty"]
            universe = features.query("liquid").sort_values("universe_score", ascending=False)
            display(universe)
            universe[["adv_usd", "momentum_126d", "volatility_63d"]].plot(kind="bar", subplots=True, layout=(1, 3), figsize=(15, 4), title="Universe diagnostics")
            plt.tight_layout()
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "32_liquidity_capacity_impact.ipynb",
        "Liquidity Capacity and Market Impact",
        "Liquidity and capacity",
        "Estimates position capacity, participation limits and simple Amihud-style impact using adjusted prices and volume.",
        ["qj.eod.get_historical_prices", "qj.finra.get_short_interest"],
        [
            """
            symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "JPM"]
            prices, volumes = price_panel(symbols)
            ret = returns(prices)
            adv = dollar_adv(prices, volumes).iloc[-1]
            """,
            """
            dollar_volume = prices * volumes
            amihud = (ret.abs() / dollar_volume).rolling(63).mean().iloc[-1] * 1e9
            capacity = pd.DataFrame({
                "adv_usd": adv,
                "capacity_5pct_adv": adv * 0.05,
                "capacity_10pct_adv": adv * 0.10,
                "capacity_20pct_adv": adv * 0.20,
                "amihud_x1e9": amihud,
                "volatility_63d": ret.tail(63).std() * np.sqrt(252),
            }).sort_values("capacity_10pct_adv", ascending=False)
            display(capacity)
            """,
            """
            target_book = 50_000_000
            equal_position = target_book / len(symbols)
            capacity["target_position"] = equal_position
            capacity["days_to_trade_10pct_adv"] = capacity["target_position"] / capacity["capacity_10pct_adv"]
            capacity["capacity_flag"] = np.where(capacity["days_to_trade_10pct_adv"] > 5, "capacity constrained", "ok")
            display(capacity[["target_position", "days_to_trade_10pct_adv", "capacity_flag"]])
            capacity[["capacity_5pct_adv", "capacity_10pct_adv", "capacity_20pct_adv"]].plot(kind="bar", title="Capacity under participation limits")
            plt.ylabel("USD")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "33_stress_testing_macro_scenarios.ipynb",
        "Stress Testing Macro Scenarios",
        "Risk and scenarios",
        "Maps custom macro and market shocks onto a portfolio through observed factor betas.",
        ["qj.eod.get_historical_prices", "qj.fred.get_cpi", "qj.fred.get_effective_federal_funds_rate", "qj.cftc.get_cot_summary"],
        [
            """
            holdings = {"AAPL": 0.18, "MSFT": 0.18, "NVDA": 0.14, "GOOGL": 0.12, "AMZN": 0.12, "JPM": 0.10, "XOM": 0.08, "LLY": 0.08}
            factors = ["SPY", "TLT", "UUP", "GLD", "DBC"]
            prices, volumes = price_panel(list(holdings) + factors)
            ret = returns(prices)
            portfolio = portfolio_returns(ret, pd.Series(holdings))
            factor_ret = ret[factors]
            macro = {
                "cpi": safe_call("FRED CPI", qj.fred.get_cpi),
                "fed_funds": safe_call("FRED effective fed funds", qj.fred.get_effective_federal_funds_rate),
                "cot": safe_call("CFTC COT summary", qj.cftc.get_cot_summary, symbol="SPX"),
            }
            """,
            """
            betas = rolling_betas(portfolio, factor_ret, window=252).tail(1).T
            betas.columns = ["portfolio_beta"]
            scenarios = pd.DataFrame({
                "equity_selloff": {"SPY": -0.08, "TLT": 0.02, "UUP": 0.015, "GLD": 0.025, "DBC": -0.02},
                "rates_up": {"SPY": -0.03, "TLT": -0.07, "UUP": 0.02, "GLD": -0.01, "DBC": 0.01},
                "inflation_spike": {"SPY": -0.04, "TLT": -0.05, "UUP": 0.01, "GLD": 0.03, "DBC": 0.08},
                "risk_on": {"SPY": 0.05, "TLT": -0.01, "UUP": -0.01, "GLD": -0.01, "DBC": 0.02},
            }).T
            scenario_pnl = scenarios @ betas["portfolio_beta"]
            display(betas)
            display((scenario_pnl * 100).rename("estimated_portfolio_return_pct"))
            scenario_pnl.mul(100).plot(kind="bar", title="Scenario P&L proxy")
            plt.ylabel("%")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "34_index_replication_tracking_error.ipynb",
        "Index Replication and Tracking Error",
        "Portfolio construction",
        "Builds a constrained replication basket and measures tracking error against SPY.",
        ["qj.eod.get_historical_prices", "qj.yf.get_sp500_stocks_info"],
        [
            """
            benchmark = "SPY"
            candidates = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "JPM", "LLY", "XOM", "UNH", "V"]
            sp500_info = safe_call("S&P 500 constituents/info", qj.yf.get_sp500_stocks_info)
            prices, volumes = price_panel([benchmark] + candidates)
            ret = returns(prices).dropna()
            """,
            """
            y = ret[benchmark]
            x = ret[candidates]
            beta = np.linalg.lstsq(x.to_numpy(), y.to_numpy(), rcond=None)[0]
            weights = pd.Series(np.maximum(beta, 0), index=candidates)
            weights = weights / weights.sum()
            weights = weights.clip(upper=0.20)
            weights = weights / weights.sum()
            replica = portfolio_returns(ret, weights)
            active = replica - y.reindex(replica.index)
            tracking_error = active.std() * np.sqrt(252)
            info_ratio = active.mean() * 252 / tracking_error
            display(weights.sort_values(ascending=False).rename("replication_weight"))
            print(f"Tracking error: {tracking_error:.2%}; information ratio proxy: {info_ratio:.2f}")
            plot_nav({"replica": replica, "SPY": y}, "Index replication NAV")
            active.cumsum().plot(title="Cumulative active return")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "36_rolling_risk_budgeting_drawdown_control.ipynb",
        "Rolling Risk Budgeting and Drawdown Control",
        "Risk management",
        "Applies volatility targeting, rolling risk budgeting and drawdown exposure control to a multi-asset portfolio.",
        ["qj.eod.get_historical_prices", "qj.fred.get_effective_federal_funds_rate"],
        [
            """
            symbols = ["SPY", "TLT", "GLD", "DBC", "UUP"]
            prices, volumes = price_panel(symbols)
            ret = returns(prices).dropna()
            base_weights = inverse_vol_weights(ret)
            raw = portfolio_returns(ret, base_weights)
            """,
            """
            target_vol = 0.10
            realized_vol = raw.rolling(63).std() * np.sqrt(252)
            leverage = (target_vol / realized_vol).clip(0.25, 1.50).shift(1).fillna(1.0)
            nav_raw = (1 + raw).cumprod()
            drawdown = nav_raw / nav_raw.cummax() - 1
            dd_control = pd.Series(1.0, index=raw.index)
            dd_control[drawdown < -0.08] = 0.50
            dd_control[drawdown < -0.15] = 0.25
            managed = raw * leverage * dd_control
            summary = pd.DataFrame({"raw": performance_stats(raw), "managed": performance_stats(managed)}).T
            display(summary)
            plot_nav({"raw inverse-vol": raw, "vol-target + drawdown control": managed}, "Risk-budgeted NAV")
            pd.DataFrame({"leverage": leverage, "drawdown_control": dd_control}).plot(title="Risk controls")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "37_brinson_attribution_benchmark_relative.ipynb",
        "Brinson Benchmark-Relative Attribution",
        "Attribution",
        "Calculates allocation, selection and interaction effects using holdings, benchmark weights and sector return buckets.",
        ["qj.eod.get_historical_prices", "qj.yf.get_sp500_sectors"],
        [
            """
            fallback_sectors = {
                "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Communication",
                "AMZN": "Consumer", "JPM": "Financials", "XOM": "Energy", "LLY": "Health Care",
            }
            sector_feed = safe_call("S&P 500 sectors", qj.yf.get_sp500_sectors)

            def normalize_sector_map(payload: Any) -> dict[str, str]:
                rows = as_rows(payload)
                sector_map = {}
                for item in rows:
                    symbol = item.get("symbol") or item.get("ticker") or item.get("Symbol")
                    sector = item.get("sector") or item.get("Sector") or item.get("gics_sector")
                    if symbol and sector:
                        sector_map[str(symbol).upper()] = str(sector)
                return sector_map

            sectors = {**fallback_sectors, **normalize_sector_map(sector_feed)}
            portfolio_w = pd.Series({"AAPL": 0.18, "MSFT": 0.18, "NVDA": 0.16, "GOOGL": 0.12, "AMZN": 0.12, "JPM": 0.10, "XOM": 0.06, "LLY": 0.08})
            benchmark_w = pd.Series({"AAPL": 0.12, "MSFT": 0.12, "NVDA": 0.10, "GOOGL": 0.09, "AMZN": 0.09, "JPM": 0.08, "XOM": 0.10, "LLY": 0.08})
            benchmark_w = benchmark_w / benchmark_w.sum()
            prices, volumes = price_panel(list(portfolio_w.index))
            period_return = prices.iloc[-1] / prices.iloc[0] - 1
            """,
            """
            df = pd.DataFrame({"sector": pd.Series(sectors), "portfolio_w": portfolio_w, "benchmark_w": benchmark_w, "return": period_return})
            sector = df.groupby("sector").apply(lambda x: pd.Series({
                "portfolio_w": x["portfolio_w"].sum(),
                "benchmark_w": x["benchmark_w"].sum(),
                "portfolio_return": np.average(x["return"], weights=x["portfolio_w"] / x["portfolio_w"].sum()),
                "benchmark_return": np.average(x["return"], weights=x["benchmark_w"] / x["benchmark_w"].sum()),
            }))
            benchmark_total = float((df["benchmark_w"] * df["return"]).sum())
            sector["allocation"] = (sector["portfolio_w"] - sector["benchmark_w"]) * (sector["benchmark_return"] - benchmark_total)
            sector["selection"] = sector["benchmark_w"] * (sector["portfolio_return"] - sector["benchmark_return"])
            sector["interaction"] = (sector["portfolio_w"] - sector["benchmark_w"]) * (sector["portfolio_return"] - sector["benchmark_return"])
            display(sector)
            sector[["allocation", "selection", "interaction"]].plot(kind="bar", stacked=True, title="Brinson attribution")
            plt.ylabel("Contribution")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "38_congress_smart_money_overlay.ipynb",
        "Congress Smart-Money Overlay",
        "Alternative data / signals",
        "Combines congressional trade feeds with price reactions to create a transparent event overlay.",
        ["qj.fmp.get_senate_trades", "qj.fmp.get_house_trades", "qj.eod.get_historical_prices"],
        [
            """
            symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
            senate = safe_call("FMP senate trades", qj.fmp.get_senate_trades, symbol="AAPL")
            house = safe_call("FMP house trades", qj.fmp.get_house_trades, symbol="AAPL")
            prices, volumes = price_panel(symbols)
            ret = returns(prices)
            """,
            """
            events = pd.concat([
                pd.DataFrame(as_rows(senate)).assign(source="senate"),
                pd.DataFrame(as_rows(house)).assign(source="house"),
            ], ignore_index=True)

            def first_present(columns: list[str], candidates: list[str]) -> str | None:
                normalized = {str(col).lower(): col for col in columns}
                for candidate in candidates:
                    if candidate.lower() in normalized:
                        return normalized[candidate.lower()]
                for col in columns:
                    low = str(col).lower()
                    if any(candidate.lower() in low for candidate in candidates):
                        return col
                return None

            if not events.empty:
                date_col = first_present(list(events.columns), ["transactionDate", "transaction_date", "disclosureDate", "reportingDate", "filingDate", "date"])
                symbol_col = first_present(list(events.columns), ["symbol", "ticker", "assetTicker", "asset_ticker"])
                events["event_date"] = pd.to_datetime(events[date_col], errors="coerce") if date_col else pd.NaT
                events["symbol"] = events[symbol_col].astype(str).str.upper() if symbol_col else "AAPL"
                event_returns = []
                for row in events.dropna(subset=["event_date"]).itertuples():
                    symbol = getattr(row, "symbol", "AAPL")
                    if symbol in prices:
                        idx = prices.index.searchsorted(row.event_date)
                        if 0 <= idx < len(prices) - 21:
                            event_returns.append({"symbol": symbol, "event_date": row.event_date, "fwd_21d": prices[symbol].iloc[idx + 21] / prices[symbol].iloc[idx] - 1})
                event_returns = pd.DataFrame(event_returns)
            else:
                event_returns = pd.DataFrame(columns=["symbol", "event_date", "fwd_21d"])
            display(events.head())
            display(event_returns.describe(include="all"))
            prices.div(prices.iloc[0]).plot(title="Price context for congress-trade overlay")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "39_institutional_crowding_13f_flows.ipynb",
        "Institutional Crowding and 13F Flow Signals",
        "Regulatory / crowding",
        "Uses SEC/FMP institutional ownership calls with returns data to build crowding, concentration and flow proxies.",
        ["qj.sec.get_institutional_holdings", "qj.fmp.get_institutional_holders", "qj.eod.get_historical_prices"],
        [
            """
            symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
            ownership = {
                symbol: safe_call(f"SEC institutional holdings {symbol}", qj.sec.get_institutional_holdings, symbol=symbol)
                for symbol in symbols[:3]
            }
            fmp_holders = {
                symbol: safe_call(f"FMP institutional holders {symbol}", qj.fmp.get_institutional_holders, symbol=symbol)
                for symbol in symbols[:3]
            }
            prices, volumes = price_panel(symbols)
            ret = returns(prices)
            """,
            """
            rows = []
            for symbol, payload in fmp_holders.items():
                holder_rows = as_rows(payload)
                values = []
                for item in holder_rows:
                    value = item.get("value") or item.get("marketValue") or item.get("shares")
                    values.append(pd.to_numeric(value, errors="coerce"))
                values = pd.Series(values).dropna()
                rows.append({
                    "symbol": symbol,
                    "holder_count": len(holder_rows),
                    "top10_concentration": values.nlargest(10).sum() / values.sum() if values.sum() else np.nan,
                })
            crowding = pd.DataFrame(rows).set_index("symbol")
            crowding["momentum_126d"] = prices.pct_change(126).iloc[-1].reindex(crowding.index)
            display(crowding)
            crowding[["holder_count", "top10_concentration", "momentum_126d"]].plot(kind="bar", subplots=True, layout=(1, 3), figsize=(15, 4), title="Crowding diagnostics")
            plt.tight_layout()
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "41_broad_event_study_pead.ipynb",
        "Broad Earnings Event Study and PEAD",
        "Event studies",
        "Extends earnings analysis into a multi-name post-earnings drift workflow with event windows and sector splits.",
        ["qj.fmp.get_earnings_calendar", "qj.fmp.get_earnings_surprises", "qj.eod.get_historical_prices"],
        [
            """
            symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
            calendar = safe_call("FMP earnings calendar", qj.fmp.get_earnings_calendar, from_date="2024-01-01", to_date=END)
            surprises = {symbol: safe_call(f"FMP earnings surprises {symbol}", qj.fmp.get_earnings_surprises, symbol=symbol) for symbol in symbols}
            prices, volumes = price_panel(symbols, start="2023-01-01", end=END)
            """,
            """
            event_rows = []
            for symbol, payload in surprises.items():
                for item in as_rows(payload):
                    event_date = pd.to_datetime(item.get("date") or item.get("fiscalDateEnding"), errors="coerce")
                    surprise = pd.to_numeric(item.get("surprisePercentage") or item.get("surprise"), errors="coerce")
                    if pd.notna(event_date):
                        event_rows.append({"symbol": symbol, "event_date": event_date, "surprise": surprise})
            events = pd.DataFrame(event_rows)
            if events.empty:
                events = pd.DataFrame({"symbol": symbols, "event_date": [prices.index[-126]] * len(symbols), "surprise": np.nan})
            curves = []
            for row in events.dropna(subset=["event_date"]).itertuples():
                if row.symbol not in prices:
                    continue
                idx = prices.index.searchsorted(row.event_date)
                if idx < 21 or idx + 42 >= len(prices):
                    continue
                window = prices[row.symbol].iloc[idx - 10:idx + 43]
                curve = window / window.iloc[10] - 1
                curves.append(pd.Series(curve.values, index=range(-10, len(curve) - 10), name=row.symbol))
            event_curve = pd.concat(curves, axis=1) if curves else pd.DataFrame()
            display(events.head())
            if not event_curve.empty:
                event_curve.mean(axis=1).plot(title="Average post-earnings drift curve")
                plt.axvline(0, color="black", linestyle="--")
                plt.ylabel("Return vs event day")
                plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "42_factor_risk_model_construction.ipynb",
        "Factor Risk Model Construction",
        "Risk models",
        "Builds sample, shrinkage and factor-model covariance estimates from price and factor returns.",
        ["qj.eod.get_historical_prices", "qj.ff.get_factors"],
        [
            """
            assets = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
            factor_symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
            ff_factors = safe_call("Fama-French factors", qj.ff.get_factors, region="US")
            prices, volumes = price_panel(assets + factor_symbols)
            ret = returns(prices).dropna()
            asset_ret = ret[assets]
            factor_ret = ret[factor_symbols]
            """,
            """
            sample_cov = asset_ret.cov() * 252
            diagonal = pd.DataFrame(np.diag(np.diag(sample_cov)), index=assets, columns=assets)
            shrinkage_cov = 0.75 * sample_cov + 0.25 * diagonal
            betas = pd.DataFrame({asset: rolling_betas(asset_ret[asset], factor_ret, window=252).iloc[-1] for asset in assets}).T
            factor_cov = factor_ret.cov() * 252
            systematic = betas @ factor_cov @ betas.T
            resid = asset_ret - factor_ret @ betas.T
            specific = pd.DataFrame(np.diag(resid.var() * 252), index=assets, columns=assets)
            factor_model_cov = systematic + specific
            display(betas)
            display(pd.DataFrame({
                "sample_vol": np.sqrt(np.diag(sample_cov)),
                "shrinkage_vol": np.sqrt(np.diag(shrinkage_cov)),
                "factor_model_vol": np.sqrt(np.diag(factor_model_cov)),
            }, index=assets))
            plt.imshow(factor_model_cov.corr(), cmap="RdBu", vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(assets)), assets, rotation=45)
            plt.yticks(range(len(assets)), assets)
            plt.title("Factor-model covariance correlation")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "44_cta_futures_carry_trend_macro.ipynb",
        "CTA Futures Carry, Trend and Macro Overlay",
        "CTA / futures",
        "Creates a simple CTA-like signal book using trend proxies, optional COT positioning and macro context.",
        ["qj.eod.get_historical_prices", "qj.eod.get_futures_pricing", "qj.cftc.get_cot_summary", "qj.fred.get_effective_federal_funds_rate"],
        [
            """
            proxies = ["DBC", "GLD", "TLT", "UUP", "SPY"]
            cot = safe_call("CFTC COT summary", qj.cftc.get_cot_summary, symbol="GC")
            futures = safe_call("EOD futures pricing", qj.eod.get_futures_pricing, symbol="CL")
            fed = safe_call("FRED fed funds", qj.fred.get_effective_federal_funds_rate)
            prices, volumes = price_panel(proxies)
            ret = returns(prices)
            """,
            """
            trend_63 = prices.pct_change(63)
            trend_252 = prices.pct_change(252)
            vol = ret.rolling(63).std() * np.sqrt(252)
            signal = (0.5 * np.sign(trend_63) + 0.5 * np.sign(trend_252)).shift(1).fillna(0)
            scaled = signal.div(vol).replace([np.inf, -np.inf], np.nan).fillna(0)
            weights = scaled.div(scaled.abs().sum(axis=1), axis=0).fillna(0)
            cta_ret = (weights.shift(1) * ret).sum(axis=1)
            display(weights.tail())
            plot_nav({"CTA trend proxy": cta_ret, "SPY": ret["SPY"]}, "CTA trend proxy vs SPY")
            weights.tail(252).plot(title="CTA proxy weights")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "45_holdings_based_vs_returns_based.ipynb",
        "Holdings-Based vs Returns-Based Analysis",
        "Attribution / ownership",
        "Compares 13F/holdings-style concentration with returns-based factor exposure analysis.",
        ["qj.sec.get_institutional_portfolio_summary", "qj.fmp.get_institutional_portfolio", "qj.eod.get_historical_prices"],
        [
            """
            portfolio_symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM"]
            factors = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
            sec_summary = safe_call("SEC institutional portfolio summary", qj.sec.get_institutional_portfolio_summary, cik="0001067983")
            fmp_portfolio = safe_call("FMP institutional portfolio", qj.fmp.get_institutional_portfolio, cik="0001067983")
            prices, volumes = price_panel(portfolio_symbols + factors)
            ret = returns(prices).dropna()
            weights = pd.Series(1 / len(portfolio_symbols), index=portfolio_symbols)
            port_ret = portfolio_returns(ret, weights)
            """,
            """
            betas = rolling_betas(port_ret, ret[factors], window=252)
            concentration = pd.Series({
                "top_1_weight": weights.max(),
                "top_3_weight": weights.nlargest(3).sum(),
                "hhi": float((weights ** 2).sum()),
                "effective_names": float(1 / (weights ** 2).sum()),
            })
            display(concentration)
            display(betas.tail())
            betas.plot(title="Returns-based rolling exposure")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "46_options_overlay_strategies.ipynb",
        "Options Overlay Strategies",
        "Options / overlays",
        "Uses option-chain, VIX/SKEW and underlying returns to compare covered-call, put-write and collar payoff profiles.",
        ["qj.cboe.get_options_expirations", "qj.cboe.get_options_chain", "qj.cboe.get_vix_data", "qj.cboe.get_skew_index_data", "qj.eod.get_historical_prices"],
        [
            """
            symbol = "SPY"
            expirations = safe_call("CBOE expirations", qj.cboe.get_options_expirations, symbol=symbol)
            chain = safe_call("CBOE options chain", qj.cboe.get_options_chain, symbol=symbol)
            vix = safe_call("CBOE VIX", qj.cboe.get_vix_data, start_date=START, end_date=END)
            skew = safe_call("CBOE SKEW", qj.cboe.get_skew_index_data, start_date=START, end_date=END)
            prices, volumes = price_panel([symbol])
            ret = returns(prices)[symbol]
            """,
            """
            monthly = ret.resample("M").sum()
            covered_call = monthly.clip(upper=0.03) + 0.006
            put_write = monthly.clip(lower=-0.05) + 0.004
            collar = monthly.clip(lower=-0.04, upper=0.025) + 0.001
            overlays = pd.DataFrame({"underlying": monthly, "covered_call": covered_call, "put_write": put_write, "collar": collar}).dropna()
            display(overlays.apply(performance_stats).T)
            plot_nav({col: overlays[col] for col in overlays.columns}, "Options overlay payoff proxies")
            overlays.cumsum().plot(title="Cumulative monthly overlay returns")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "47_tactical_asset_allocation_macro_valuation.ipynb",
        "Tactical Asset Allocation with Macro and Valuation",
        "Asset allocation",
        "Combines macro, valuation and sentiment feeds with market data to create a dynamic allocation rule.",
        ["qj.fred.get_cpi", "qj.fred.get_effective_federal_funds_rate", "qj.multpl.get_shiller_pe_ratio", "qj.cnnf.get_latest_fear_greed_value", "qj.eod.get_historical_prices"],
        [
            """
            assets = ["SPY", "TLT", "GLD", "DBC", "UUP"]
            macro = {
                "cpi": safe_call("FRED CPI", qj.fred.get_cpi),
                "fed_funds": safe_call("FRED fed funds", qj.fred.get_effective_federal_funds_rate),
                "shiller_pe": safe_call("Multpl Shiller PE", qj.multpl.get_shiller_pe_ratio),
                "fear_greed": safe_call("CNN fear & greed", qj.cnnf.get_latest_fear_greed_value),
            }
            prices, volumes = price_panel(assets)
            ret = returns(prices)
            """,
            """
            momentum = prices.pct_change(126)
            vol = ret.rolling(63).std() * np.sqrt(252)
            raw_score = momentum.rank(axis=1, pct=True) - vol.rank(axis=1, pct=True) * 0.35
            weights = raw_score.clip(lower=0).div(raw_score.clip(lower=0).sum(axis=1), axis=0).fillna(0)
            weights = weights.rolling(21).mean().fillna(method="bfill")
            taa = (weights.shift(1) * ret).sum(axis=1)
            equal = ret.mean(axis=1)
            display(weights.tail())
            plot_nav({"macro/valuation TAA proxy": taa, "equal-weight": equal}, "Tactical allocation proxy")
            weights.tail(504).plot(title="Allocation weights")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "48_end_to_end_research_to_book.ipynb",
        "End-to-End Research to Book",
        "End-to-end workflow",
        "Runs universe scoring, weight construction, risk attribution, drawdown review and report-ready outputs in one notebook.",
        ["qj.eod.get_historical_prices", "qj.fmp.get_financial_ratios_ttm", "qj.finra.get_short_interest", "qj.insider.latest_trades"],
        [
            """
            universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "JPM", "LLY", "XOM"]
            prices, volumes = price_panel(universe)
            ret = returns(prices)
            insider = safe_call("Insider latest trades", qj.insider.latest_trades, symbols=universe[:5])
            """,
            """
            ratios = []
            for symbol in universe:
                data = unwrap(safe_call(f"FMP ratios {symbol}", qj.fmp.get_financial_ratios_ttm, symbol=symbol)) or {}
                ratios.append({"symbol": symbol, "gross_margin": pd.to_numeric(data.get("grossProfitMarginTTM"), errors="coerce"), "pe": pd.to_numeric(data.get("peRatioTTM"), errors="coerce")})
            ratios = pd.DataFrame(ratios).set_index("symbol")
            score = (
                prices.pct_change(126).iloc[-1].rank(pct=True)
                - ret.tail(63).std().rank(pct=True)
                + ratios["gross_margin"].rank(pct=True).reindex(universe).fillna(0)
                - ratios["pe"].rank(pct=True).reindex(universe).fillna(0) * 0.35
            )
            raw = score.clip(lower=0)
            weights = raw / raw.sum()
            book_ret = portfolio_returns(ret, weights)
            report = pd.DataFrame({
                "weight": weights,
                "risk_contribution": risk_contribution(ret, weights),
                "momentum_126d": prices.pct_change(126).iloc[-1],
                "vol_63d": ret.tail(63).std() * np.sqrt(252),
            }).sort_values("weight", ascending=False)
            display(report)
            display(performance_stats(book_ret))
            plot_nav({"candidate book": book_ret, "equal universe": ret[universe].mean(axis=1)}, "Research-to-book NAV")
            report[["weight", "risk_contribution"]].plot(kind="bar", title="Weight vs risk contribution")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "50_integrated_daily_risk_attribution_report.ipynb",
        "Integrated Daily Risk Attribution Report",
        "Daily PM report",
        "Produces a daily intelligence packet from holdings, factor moves, earnings/filings/insider context and risk contributors.",
        ["qj.eod.get_historical_prices", "qj.sec.get_recent_filings", "qj.insider.latest_trades", "qj.fmp.get_earnings_calendar"],
        [
            """
            holdings = pd.Series({"AAPL": 0.18, "MSFT": 0.18, "NVDA": 0.16, "GOOGL": 0.12, "AMZN": 0.12, "JPM": 0.10, "XOM": 0.06, "LLY": 0.08})
            factors = ["SPY", "QQQ", "TLT", "GLD", "UUP"]
            prices, volumes = price_panel(list(holdings.index) + factors)
            ret = returns(prices)
            filings = safe_call("SEC recent filings", qj.sec.get_recent_filings, limit=20)
            insider = safe_call("Insider latest trades", qj.insider.latest_trades, symbols=list(holdings.index))
            earnings = safe_call("FMP earnings calendar", qj.fmp.get_earnings_calendar, from_date=END, to_date=END)
            """,
            """
            latest_returns = ret[list(holdings.index)].iloc[-1]
            contribution = holdings * latest_returns
            book_ret = portfolio_returns(ret, holdings)
            betas = rolling_betas(book_ret, ret[factors], window=126).tail(1).T
            risk = risk_contribution(ret[list(holdings.index)], holdings)
            report = pd.DataFrame({
                "weight": holdings,
                "latest_return": latest_returns,
                "daily_contribution": contribution,
                "risk_contribution": risk,
            }).sort_values("daily_contribution")
            display(report)
            display(betas.rename(columns={betas.columns[0]: "latest_beta"}) if not betas.empty else betas)
            report[["daily_contribution", "risk_contribution"]].plot(kind="bar", title="Daily contribution and risk")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "52_turnover_cost_drag.ipynb",
        "Turnover and Cost Drag",
        "Execution-aware backtesting",
        "Quantifies how turnover, slippage assumptions and rebalance frequency change an otherwise attractive signal.",
        ["qj.eod.get_historical_prices"],
        [
            """
            symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "JPM"]
            prices, volumes = price_panel(symbols)
            ret = returns(prices)
            """,
            """
            def momentum_weights(prices: pd.DataFrame, lookback: int = 126, top_n: int = 4) -> pd.DataFrame:
                signal = prices.pct_change(lookback)
                weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
                for dt, row in signal.iterrows():
                    winners = row.nlargest(top_n).dropna().index
                    if len(winners):
                        weights.loc[dt, winners] = 1 / len(winners)
                return weights

            weights = momentum_weights(prices).resample("M").last().reindex(prices.index).ffill().fillna(0)
            gross = (weights.shift(1) * ret).sum(axis=1)
            turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
            results = {}
            for bps in [0, 5, 10, 25, 50]:
                net = gross - turnover * (bps / 10000)
                results[f"{bps} bps"] = net
            summary = pd.DataFrame({name: performance_stats(series) for name, series in results.items()}).T
            display(summary)
            plot_nav(results, "Turnover cost drag")
            turnover.rolling(21).sum().plot(title="Rolling turnover")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "53_backtest_with_risk_management_full.ipynb",
        "Backtest with Risk Management",
        "Backtesting",
        "Compares a raw momentum strategy with a risk-managed variant using volatility targeting, drawdown controls and concentration limits.",
        ["qj.eod.get_historical_prices", "qj.bt.prepare", "qj.analytics.hv"],
        [
            """
            symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
            bt_prepare = safe_call("BT prepare metadata", qj.bt.prepare, symbols=symbols, start=START, end=END)
            hv = safe_call("Analytics historical volatility", qj.analytics.hv, symbol="SPY", start=START, end=END)
            prices, volumes = price_panel(symbols)
            ret = returns(prices)
            """,
            """
            signal = prices.pct_change(126).rank(axis=1, pct=True)
            raw_weights = signal.clip(lower=0).div(signal.clip(lower=0).sum(axis=1), axis=0).fillna(0)
            raw_weights = raw_weights.clip(upper=0.35)
            raw_weights = raw_weights.div(raw_weights.sum(axis=1), axis=0).fillna(0)
            raw_ret = (raw_weights.shift(1) * ret).sum(axis=1)
            vol = raw_ret.rolling(63).std() * np.sqrt(252)
            scale = (0.12 / vol).clip(0.25, 1.25).shift(1).fillna(1.0)
            nav = (1 + raw_ret).cumprod()
            dd = nav / nav.cummax() - 1
            managed_ret = raw_ret * scale * np.where(dd < -0.10, 0.5, 1.0)
            summary = pd.DataFrame({"raw": performance_stats(raw_ret), "risk_managed": performance_stats(managed_ret)}).T
            display(summary)
            plot_nav({"raw momentum": raw_ret, "risk-managed": managed_ret}, "Backtest with risk management")
            pd.DataFrame({"scale": scale, "drawdown": dd}).plot(subplots=True, title="Risk controls")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "55_factor_timing_dynamic_exposures.ipynb",
        "Factor Timing and Dynamic Exposures",
        "Factor timing",
        "Uses macro regimes and factor proxies to time equity, growth, small-cap, duration and gold exposures.",
        ["qj.eod.get_historical_prices", "qj.ff.get_factors", "qj.fred.get_cpi", "qj.fred.get_effective_federal_funds_rate"],
        [
            """
            factors = ["SPY", "QQQ", "IWM", "TLT", "GLD", "DBC"]
            ff = safe_call("Fama-French factors", qj.ff.get_factors, region="US")
            cpi = safe_call("FRED CPI", qj.fred.get_cpi)
            fed = safe_call("FRED fed funds", qj.fred.get_effective_federal_funds_rate)
            prices, volumes = price_panel(factors)
            ret = returns(prices)
            """,
            """
            momentum = prices.pct_change(126)
            trend = zscore(momentum.mean(axis=1), 252)
            inflation_proxy = ret["DBC"].rolling(63).sum() - ret["TLT"].rolling(63).sum()
            regime_risk_on = (trend > 0).astype(float)
            weights = pd.DataFrame(index=ret.index, columns=factors, dtype=float)
            weights["SPY"] = 0.25 + 0.20 * regime_risk_on
            weights["QQQ"] = 0.20 + 0.15 * regime_risk_on
            weights["IWM"] = 0.15 * regime_risk_on
            weights["TLT"] = 0.25 - 0.10 * regime_risk_on - 0.10 * (inflation_proxy > 0).astype(float)
            weights["GLD"] = 0.15 + 0.10 * (inflation_proxy > 0).astype(float)
            weights = weights.clip(lower=0).div(weights.sum(axis=1), axis=0).fillna(method="bfill")
            dynamic = (weights.shift(1) * ret).sum(axis=1)
            equal = ret.mean(axis=1)
            display(weights.tail())
            plot_nav({"dynamic factor timing": dynamic, "equal factors": equal}, "Dynamic factor timing")
            weights.tail(504).plot(title="Dynamic exposures")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "57_crypto_perps_basis_funding_arbitrage.ipynb",
        "Crypto Perps Basis and Funding Arbitrage",
        "Crypto / derivatives",
        "Pulls CCXT-style spot/perp data and funding history where available, then builds funding and basis diagnostics.",
        ["qj.ccxt.get_historical_prices", "qj.ccxt.get_historical_funding_rates", "qj.ccxt.get_open_interest", "qj.coingecko.get_historical_prices"],
        [
            """
            btc_spot = safe_call("CCXT BTC spot OHLCV", qj.ccxt.get_historical_prices, symbol="BTC/USDT", exchange="binance", timeframe="1d", since=START)
            funding = safe_call("CCXT BTC funding", qj.ccxt.get_historical_funding_rates, symbol="BTC/USDT", exchange="binance")
            open_interest = safe_call("CCXT BTC open interest", qj.ccxt.get_open_interest, symbol="BTC/USDT", exchange="binance")
            gecko = safe_call("CoinGecko BTC historical", qj.coingecko.get_historical_prices, coin_id="bitcoin", vs_currency="usd", days="max")
            """,
            """
            rows = as_rows(btc_spot) or as_rows(gecko)
            prices_df = pd.DataFrame(rows)
            if not prices_df.empty:
                date_col = "date" if "date" in prices_df else prices_df.columns[0]
                numeric_cols = prices_df.select_dtypes(include="number").columns
                value_col = "close" if "close" in prices_df else (numeric_cols[-1] if len(numeric_cols) else prices_df.columns[-1])
                prices_df["date"] = pd.to_datetime(prices_df[date_col], errors="coerce")
                prices_df["price"] = pd.to_numeric(prices_df[value_col], errors="coerce")
                prices_df = prices_df.dropna(subset=["date", "price"]).set_index("date").sort_index()
                btc_ret = prices_df["price"].pct_change().dropna()
            else:
                btc_ret = pd.Series(dtype=float)
            funding_df = pd.DataFrame(as_rows(funding))
            display(funding_df.head())
            if not btc_ret.empty:
                nav = (1 + btc_ret).cumprod()
                nav.plot(title="BTC spot context for funding/basis research")
                plt.show()
                display(performance_stats(btc_ret))
            """,
        ],
    ),
]


ADVANCED_FULL_SPECS = [
    NotebookSpec(
        "61_vectorized_strategy_grid.ipynb",
        "Vectorized Strategy Grid",
        "Advanced buy-side diagnostics",
        "Runs a complete SMA parameter sweep from live QuantJourney price data, including transaction costs and robustness surfaces.",
        ["qj.eod.get_historical_prices"],
        [
            """
            prices, volumes = price_panel(["SPY"])
            spy = prices["SPY"].dropna()
            ret = spy.pct_change().fillna(0)
            fast_windows = [5, 10, 15, 20, 30, 40, 50]
            slow_windows = [60, 80, 100, 125, 150, 200]
            """,
            """
            def sma_strategy(price: pd.Series, fast: int, slow: int, cost_bps: float = 5.0) -> pd.Series:
                fast_ma = price.rolling(fast).mean()
                slow_ma = price.rolling(slow).mean()
                position = (fast_ma > slow_ma).astype(float).shift(1).fillna(0)
                turnover = position.diff().abs().fillna(position.abs())
                return position * price.pct_change().fillna(0) - turnover * cost_bps / 10000

            sharpe = pd.DataFrame(index=fast_windows, columns=slow_windows, dtype=float)
            total_return = sharpe.copy()
            max_dd = sharpe.copy()
            strategies = {}
            for fast in fast_windows:
                for slow in slow_windows:
                    if fast >= slow:
                        continue
                    sret = sma_strategy(spy, fast, slow)
                    strategies[(fast, slow)] = sret
                    st = performance_stats(sret)
                    sharpe.loc[fast, slow] = st["sharpe"]
                    total_return.loc[fast, slow] = st["total_return"]
                    max_dd.loc[fast, slow] = st["max_drawdown"]
            best = sharpe.stack().idxmax()
            print("Best fast/slow:", best)
            display(sharpe)
            """,
            """
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            for ax, data, title in zip(axes, [sharpe, total_return, max_dd], ["Sharpe", "Total return", "Max drawdown"]):
                im = ax.imshow(data.astype(float), aspect="auto")
                ax.set_xticks(range(len(data.columns)), data.columns)
                ax.set_yticks(range(len(data.index)), data.index)
                ax.set_title(title)
                fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.show()
            plot_nav({f"SMA {best[0]}/{best[1]}": strategies[best], "SPY": ret}, "Best grid strategy vs SPY")
            """,
        ],
        mirror_to_advanced=True,
    ),
    NotebookSpec(
        "62_walk_forward_robustness.ipynb",
        "Walk-Forward Robustness",
        "Advanced buy-side diagnostics",
        "Selects parameters on rolling training windows and evaluates the next out-of-sample fold.",
        ["qj.eod.get_historical_prices"],
        [
            """
            prices, volumes = price_panel(["SPY"])
            spy = prices["SPY"].dropna()
            fast_windows = [5, 10, 15, 20, 30, 40, 50]
            slow_windows = [60, 80, 100, 125, 150, 200]

            def sma_strategy(price: pd.Series, fast: int, slow: int, cost_bps: float = 5.0) -> pd.Series:
                fast_ma = price.rolling(fast).mean()
                slow_ma = price.rolling(slow).mean()
                position = (fast_ma > slow_ma).astype(float).shift(1).fillna(0)
                turnover = position.diff().abs().fillna(position.abs())
                return position * price.pct_change().fillna(0) - turnover * cost_bps / 10000
            """,
            """
            strategy_map = {(f, s): sma_strategy(spy, f, s) for f in fast_windows for s in slow_windows if f < s}
            train_days = 504
            test_days = 126
            records = []
            oos = []
            for fold, start_idx in enumerate(range(train_days, len(spy) - test_days, test_days), start=1):
                train_idx = spy.index[start_idx - train_days:start_idx]
                test_idx = spy.index[start_idx:start_idx + test_days]
                train_scores = {k: performance_stats(v.reindex(train_idx).fillna(0))["sharpe"] for k, v in strategy_map.items()}
                best = max(train_scores, key=lambda k: -np.inf if pd.isna(train_scores[k]) else train_scores[k])
                test_ret = strategy_map[best].reindex(test_idx).fillna(0)
                oos.append(test_ret)
                records.append({"fold": fold, "best_fast": best[0], "best_slow": best[1], "train_sharpe": train_scores[best], "test_sharpe": performance_stats(test_ret)["sharpe"], "test_return": (1 + test_ret).prod() - 1})
            wf = pd.DataFrame(records)
            oos_ret = pd.concat(oos).sort_index()
            display(wf)
            """,
            """
            wf[["train_sharpe", "test_sharpe"]].plot(kind="bar", title="Walk-forward train/test Sharpe")
            plt.show()
            plot_nav({"walk-forward": oos_ret, "SPY": spy.pct_change().reindex(oos_ret.index).fillna(0)}, "Walk-forward out-of-sample NAV")
            """,
        ],
        mirror_to_advanced=True,
    ),
    NotebookSpec(
        "63_monte_carlo_tail_risk.ipynb",
        "Monte Carlo Tail Risk",
        "Advanced buy-side diagnostics",
        "Bootstraps portfolio returns into one-year fan charts and terminal-return tail distributions.",
        ["qj.eod.get_historical_prices"],
        [
            """
            symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
            prices, volumes = price_panel(symbols)
            ret = returns(prices).dropna()
            weights = pd.Series(1 / len(symbols), index=symbols)
            port = portfolio_returns(ret, weights)
            """,
            """
            rng = np.random.default_rng(42)
            horizon = 252
            paths = 1000
            samples = rng.choice(port.dropna().to_numpy(), size=(horizon, paths), replace=True)
            sim = np.cumprod(1 + samples, axis=0)
            bands = pd.DataFrame(np.percentile(sim, [5, 25, 50, 75, 95], axis=1).T, columns=["p05", "p25", "p50", "p75", "p95"])
            terminal = pd.Series(sim[-1] - 1)
            display(terminal.describe(percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]))
            """,
            """
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes[0].fill_between(bands.index, bands["p05"], bands["p95"], alpha=0.15, label="5-95%")
            axes[0].fill_between(bands.index, bands["p25"], bands["p75"], alpha=0.25, label="25-75%")
            axes[0].plot(bands["p50"], label="median")
            axes[0].set_title("Monte Carlo fan")
            axes[0].legend()
            terminal.mul(100).hist(ax=axes[1], bins=40)
            axes[1].set_title("Terminal return distribution")
            plt.show()
            """,
        ],
        mirror_to_advanced=True,
    ),
    NotebookSpec(
        "64_correlation_regime_lab.ipynb",
        "Correlation Regime Lab",
        "Advanced buy-side diagnostics",
        "Calculates recent cross-asset correlation, rolling pairwise regimes and drawdown context.",
        ["qj.eod.get_historical_prices"],
        [
            """
            symbols = ["SPY", "TLT", "GLD", "DBC", "UUP"]
            prices, volumes = price_panel(symbols)
            ret = returns(prices).dropna()
            recent_corr = ret.tail(252).corr()
            rolling_avg = ret.rolling(63).corr().groupby(level=0).apply(lambda x: x.where(np.triu(np.ones(x.shape), 1).astype(bool)).stack().mean())
            spy_nav = (1 + ret["SPY"]).cumprod()
            spy_dd = spy_nav / spy_nav.cummax() - 1
            display(recent_corr)
            """,
            """
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            im = axes[0].imshow(recent_corr, vmin=-1, vmax=1, cmap="RdBu")
            axes[0].set_xticks(range(len(symbols)), symbols, rotation=45)
            axes[0].set_yticks(range(len(symbols)), symbols)
            axes[0].set_title("Recent correlation")
            fig.colorbar(im, ax=axes[0])
            rolling_avg.plot(ax=axes[1], title="Rolling average pairwise correlation")
            spy_dd.mul(100).plot(ax=axes[2], title="SPY drawdown context")
            plt.tight_layout()
            plt.show()
            """,
        ],
        mirror_to_advanced=True,
    ),
    NotebookSpec(
        "65_drawdown_diagnostics.ipynb",
        "Drawdown Diagnostics",
        "Advanced buy-side diagnostics",
        "Computes SMA 5/125 equity, underwater drawdown, rolling volatility and exposure state from live SPY prices.",
        ["qj.eod.get_historical_prices"],
        [
            """
            prices, volumes = price_panel(["SPY"])
            spy = prices["SPY"].dropna()
            fast, slow = 5, 125
            fast_ma = spy.rolling(fast).mean()
            slow_ma = spy.rolling(slow).mean()
            position = (fast_ma > slow_ma).astype(float).shift(1).fillna(0)
            turnover = position.diff().abs().fillna(position.abs())
            strategy_ret = position * spy.pct_change().fillna(0) - turnover * 0.0005
            buy_hold = spy.pct_change().fillna(0)
            """,
            """
            strategy_nav = (1 + strategy_ret).cumprod()
            benchmark_nav = (1 + buy_hold).cumprod()
            drawdown = strategy_nav / strategy_nav.cummax() - 1
            rolling_vol = strategy_ret.rolling(63).std() * np.sqrt(252)
            display(pd.DataFrame({"strategy": performance_stats(strategy_ret), "buy_hold": performance_stats(buy_hold)}).T)
            """,
            """
            fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
            strategy_nav.plot(ax=axes[0], label="SMA 5/125")
            benchmark_nav.plot(ax=axes[0], label="buy & hold")
            axes[0].legend()
            axes[0].set_title("Strategy equity vs benchmark")
            drawdown.mul(100).plot(ax=axes[1], title="Underwater drawdown")
            rolling_vol.mul(100).plot(ax=axes[2], title="Rolling 63D volatility")
            position.plot(ax=axes[3], title="Exposure state")
            plt.tight_layout()
            plt.show()
            """,
        ],
        mirror_to_advanced=True,
    ),
    NotebookSpec(
        "66_factor_exposure_diagnostics.ipynb",
        "Factor Exposure Diagnostics",
        "Advanced buy-side diagnostics",
        "Estimates rolling 126-day factor betas and recent contribution proxy for a mega-cap equity basket.",
        ["qj.eod.get_historical_prices"],
        [
            """
            holdings = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
            factors = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
            prices, volumes = price_panel(holdings + factors)
            ret = returns(prices).dropna()
            weights = pd.Series(1 / len(holdings), index=holdings)
            port = portfolio_returns(ret, weights)
            betas = rolling_betas(port, ret[factors], window=126)
            """,
            """
            latest_beta = betas.iloc[-1] if not betas.empty else pd.Series(dtype=float)
            factor_quarter = ret[factors].tail(63).sum()
            contribution_proxy = latest_beta * factor_quarter
            display(latest_beta.rename("latest_beta"))
            display(contribution_proxy.rename("quarter_contribution_proxy"))
            """,
            """
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            betas.plot(ax=axes[0], title="Rolling 126D betas")
            latest_beta.plot(kind="bar", ax=axes[1], title="Latest exposure")
            contribution_proxy.mul(100).plot(kind="bar", ax=axes[2], title="Quarter contribution proxy")
            plt.tight_layout()
            plt.show()
            """,
        ],
        mirror_to_advanced=True,
    ),
]


from multisource_candidate_specs import get_multisource_workflow_specs


MULTI_SOURCE_WORKFLOW_SPECS = get_multisource_workflow_specs(NotebookSpec)
ALL_GENERATED_SPECS = [*SPECS, *ADVANCED_FULL_SPECS, *MULTI_SOURCE_WORKFLOW_SPECS]


def clean_candidates() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    for path in CANDIDATES.glob("*.ipynb"):
        path.unlink()
    index = CANDIDATES / "INDEX.md"
    if index.exists():
        index.unlink()


def copy_existing() -> list[tuple[str, str, str]]:
    copied: list[tuple[str, str, str]] = []
    for source_dir, names, category, note in [
        (ROOT / "notebooks" / "core", EXISTING_CORE, "Existing core notebook", "copied from notebooks/core"),
        (ROOT / "notebooks" / "buy_side", EXISTING_BUY_SIDE, "Existing executed buy-side notebook", "copied from notebooks/buy_side"),
    ]:
        for name in names:
            src = source_dir / name
            dst = CANDIDATES / name
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, dst)
            copied.append((name, category, note))
    return copied


def write_generated(specs: Iterable[NotebookSpec]) -> list[str]:
    written: list[str] = []
    for spec in specs:
        nb = notebook(spec)
        out = CANDIDATES / spec.filename
        out.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
        written.append(spec.filename)
        if spec.mirror_to_advanced:
            ADVANCED.mkdir(parents=True, exist_ok=True)
            (ADVANCED / spec.filename).write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
    return written


def preview_cell(name: str) -> dict:
    stem = Path(name).stem
    return md(f"""
    ## Run Output

    ![{stem}](../plots/{stem}_output_01.png)
    """)


def branding_cell() -> dict:
    return md("""
    **Prepared by QuantJourney.** Candidate notebook source is kept clean and unexecuted. Generated run artifacts are committed under `plots/` and indexed in `plots/manifest.json`.
    """)


def attach_preview_cells() -> None:
    for path in sorted(CANDIDATES.glob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        nb["cells"].insert(1, preview_cell(path.name))
        nb["cells"].insert(2, branding_cell())
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
        path.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")


def write_index(copied: list[tuple[str, str, str]], generated: list[str]) -> None:
    rows = []
    for name, category, summary in copied:
        rows.append((name, category, summary))
    for spec in ALL_GENERATED_SPECS:
        rows.append((spec.filename, spec.category, spec.summary))
    rows = sorted(rows, key=lambda row: row[0])
    lines = [
        "# Buy-Side Candidate Notebook Catalog",
        "",
        "Flat candidate catalog for institutional workflows. Candidate notebooks are clean source notebooks; generated run plots are committed under `plots/` and indexed in `plots/manifest.json`. New candidates use real QuantJourney SDK connector calls plus transparent local analytics. Production systems can wrap the same workflows behind governed domain routes, tenant scopes and audit metadata.",
        "",
        "| Notebook | Preview | Category | What it shows |",
        "|---|---|---|---|",
    ]
    for name, category, summary in rows:
        stem = Path(name).stem
        lines.append(f"| [{name}]({name}) | [PNG](../plots/{stem}_output_01.png) | {category} | {summary} |")
    lines.extend(
        [
            "",
            "## Run",
            "",
            "```bash",
            "export QJ_API_KEY=\"qj_...\"",
            "jupyter lab _candidates",
            "```",
            "",
            "Candidate notebooks use optional API calls through `safe_call(...)` when a feed may depend on tenant entitlements.",
        ]
    )
    (CANDIDATES / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_manifest_for_advanced() -> None:
    if not MANIFEST.exists():
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    existing = {entry["path"]: entry for entry in manifest}
    for spec in ADVANCED_FULL_SPECS:
        path = f"notebooks/buy_side_advanced/{spec.filename}"
        nb_path = ROOT / path
        if not nb_path.exists():
            continue
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        entry = existing.get(path)
        if entry is None:
            entry = {
                "name": spec.filename,
                "group": "buy_side_advanced",
                "path": path,
                "output_dir": "outputs/buy_side_advanced",
                "images": 1,
                "cells": len(nb.get("cells", [])),
            }
            manifest.append(entry)
        else:
            entry["cells"] = len(nb.get("cells", []))
            entry["images"] = entry.get("images", 1)
            entry["output_dir"] = "outputs/buy_side_advanced"
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    clean_candidates()
    copied = copy_existing()
    generated = write_generated(ALL_GENERATED_SPECS)
    attach_preview_cells()
    write_index(copied, generated)
    update_manifest_for_advanced()
    print(f"Copied {len(copied)} existing notebooks")
    print(f"Generated {len(generated)} candidate notebooks")
    print(f"Wrote {CANDIDATES / 'INDEX.md'}")


if __name__ == "__main__":
    main()
