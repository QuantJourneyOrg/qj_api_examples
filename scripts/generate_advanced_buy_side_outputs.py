"""Generate advanced buy-side example outputs from live QuantJourney API data.

The examples are inspired by vectorized research workflows: parameter grids,
robustness surfaces, walk-forward validation, Monte Carlo tail bands, drawdown
diagnostics, correlation regimes and rolling factor exposure.

Set QJ_API_KEY before running. The script writes PNG files to
outputs/buy_side_advanced and notebook shells to notebooks/buy_side_advanced.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantjourney.sdk import QuantJourneyAPI


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "buy_side_advanced"
NOTEBOOK_DIR = ROOT / "notebooks" / "buy_side_advanced"
MANIFEST = ROOT / "outputs" / "manifest.json"

START_DATE = os.environ.get("QJ_ADVANCED_START_DATE", "2020-01-01")
END_DATE = os.environ.get("QJ_ADVANCED_END_DATE", "2026-06-06")

BG = "#051946"
PANEL = "#06142f"
GRID = "#233553"
TEXT = "#f8fafc"
MUTED = "#cbd5e1"
ACCENT = "#60a5fa"
GREEN = "#22c55e"
RED = "#ef4444"
ORANGE = "#f59e0b"
PURPLE = "#a855f7"
CYAN = "#06b6d4"
ROSE = "#f43f5e"
YELLOW = "#facc15"

plt.rcParams.update(
    {
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10.5,
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
    }
)


def value(payload: Any) -> Any:
    """Return the normalized value payload from a QuantJourney response."""
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if isinstance(payload, dict) and "value" in payload:
        return payload["value"]
    return payload


def price_frame(qj: QuantJourneyAPI, symbol: str, start: str, end: str) -> pd.DataFrame:
    raw = value(qj.eod.get_historical_prices(symbol=symbol, start_date=start, end_date=end))
    if isinstance(raw, dict):
        raw = raw.get(symbol) or raw.get(symbol.upper()) or raw.get("prices") or []
    df = pd.DataFrame(raw)
    if df.empty:
        raise RuntimeError(f"No price data returned for {symbol}")
    if "date" not in df:
        raise RuntimeError(f"Price payload for {symbol} has no date column")

    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "adjusted_close" in df and df["adjusted_close"].notna().any():
        df["price"] = df["adjusted_close"].fillna(df["close"])
    elif "close" in df:
        df["price"] = df["close"]
    else:
        raise RuntimeError(f"Price payload for {symbol} has no close field")
    return df.dropna(subset=["price"]).sort_values("date").set_index("date")


def fetch_price_panel(qj: QuantJourneyAPI, symbols: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frames[symbol] = price_frame(qj, symbol, START_DATE, END_DATE)
            print(f"Fetched {symbol}: {len(frames[symbol])} rows")
        except Exception as exc:
            print(f"Skipping {symbol}: {exc}")
    return frames


def panel(frames: dict[str, pd.DataFrame], symbols: list[str]) -> pd.DataFrame:
    missing = [symbol for symbol in symbols if symbol not in frames]
    if missing:
        raise RuntimeError(f"Missing required price frames: {', '.join(missing)}")
    return pd.DataFrame({symbol: frames[symbol]["price"] for symbol in symbols}).dropna()


def set_dark(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.65, linewidth=0.75)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def style_colorbar(cbar: Any) -> None:
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    for label in cbar.ax.get_yticklabels():
        label.set_color(MUTED)
    cbar.outline.set_edgecolor(GRID)


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.title.set_color(TEXT)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.tick_params(colors=MUTED)
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor(PANEL)
            legend.get_frame().set_edgecolor(GRID)
            for text in legend.get_texts():
                text.set_color(TEXT)
    fig.savefig(OUTPUT_DIR / name, facecolor=BG, edgecolor=BG, dpi=170, bbox_inches="tight")
    plt.close(fig)


def strategy_returns(price: pd.Series, fast: int, slow: int, cost_bps: float = 5.0) -> pd.Series:
    returns = price.pct_change().fillna(0.0)
    fast_ma = price.rolling(fast, min_periods=fast).mean()
    slow_ma = price.rolling(slow, min_periods=slow).mean()
    signal = (fast_ma > slow_ma).astype(float)
    position = signal.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * (cost_bps / 10000.0)
    out = position * returns - costs
    out.iloc[: slow + 1] = 0.0
    return out.dropna()


def stats(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {"total_return": np.nan, "ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    equity = (1 + returns).cumprod()
    ann_return = equity.iloc[-1] ** (252 / max(len(returns), 1)) - 1 if equity.iloc[-1] > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol and np.isfinite(ann_vol) else np.nan
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(drawdown.min()),
    }


def sma_grid(price: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[int, int]]:
    fast_windows = [5, 10, 15, 20, 30, 40, 50]
    slow_windows = [60, 80, 100, 125, 150, 200]
    sharpe = pd.DataFrame(index=fast_windows, columns=slow_windows, dtype=float)
    total_return = sharpe.copy()
    max_dd = sharpe.copy()

    for fast in fast_windows:
        for slow in slow_windows:
            if fast >= slow:
                continue
            result = stats(strategy_returns(price, fast, slow))
            sharpe.loc[fast, slow] = result["sharpe"]
            total_return.loc[fast, slow] = result["total_return"]
            max_dd.loc[fast, slow] = result["max_dd"]

    best = sharpe.stack().idxmax()
    return sharpe, total_return, max_dd, (int(best[0]), int(best[1]))


def annotate_heatmap(ax: plt.Axes, data: pd.DataFrame, fmt: str, scale: float = 1.0) -> None:
    values = data.to_numpy(dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, fmt.format(values[i, j] * scale), ha="center", va="center", color=TEXT, fontsize=8)


def plot_sma_grid(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    price = frames["SPY"]["price"].dropna()
    sharpe, total_return, max_dd, best = sma_grid(price)

    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=BG)
    set_dark(ax)
    im = ax.imshow(sharpe.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(sharpe.columns)), sharpe.columns)
    ax.set_yticks(range(len(sharpe.index)), sharpe.index)
    ax.set_xlabel("Slow SMA window")
    ax.set_ylabel("Fast SMA window")
    ax.set_title("SMA Parameter Grid - Annualized Sharpe", loc="left", fontsize=17, weight="bold")
    annotate_heatmap(ax, sharpe, "{:.2f}")
    ax.scatter([list(sharpe.columns).index(best[1])], [list(sharpe.index).index(best[0])], s=190, facecolors="none", edgecolors=ROSE, linewidths=2.2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style_colorbar(cbar)
    save(fig, "advanced-sma-grid-heatmap.png")

    fig2, axs = plt.subplots(1, 3, figsize=(13.5, 5.8), facecolor=BG)
    panels = [
        (total_return * 100, "Total Return", "%", "magma"),
        (sharpe, "Sharpe", "", "viridis"),
        (max_dd * 100, "Max Drawdown", "%", "RdYlGn"),
    ]
    for ax, (data, title, suffix, cmap) in zip(axs, panels):
        set_dark(ax)
        im = ax.imshow(data.to_numpy(dtype=float), cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(data.columns)), data.columns, rotation=0)
        ax.set_yticks(range(len(data.index)), data.index)
        ax.set_title(title, loc="left", fontsize=13, weight="bold")
        if suffix == "%":
            annotate_heatmap(ax, data, "{:.0f}")
        else:
            annotate_heatmap(ax, data, "{:.2f}")
        cbar = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        style_colorbar(cbar)
    fig2.suptitle("Parameter Robustness Surface", x=0.03, y=0.98, ha="left", fontsize=17, color=TEXT, weight="bold")
    fig2.tight_layout(pad=1.8)
    save(fig2, "advanced-parameter-robustness-surface.png")

    return {"best_fast": best[0], "best_slow": best[1], "sharpe": sharpe}


def plot_walk_forward(frames: dict[str, pd.DataFrame]) -> None:
    price = frames["SPY"]["price"].dropna()
    fast_windows = [5, 10, 15, 20, 30, 40, 50]
    slow_windows = [60, 80, 100, 125, 150, 200]
    returns_by_param = {
        (fast, slow): strategy_returns(price, fast, slow)
        for fast in fast_windows
        for slow in slow_windows
        if fast < slow
    }

    records = []
    oos_parts = []
    train_len = 504
    test_len = 126
    starts = range(train_len, len(price) - test_len, test_len)

    for fold, start_idx in enumerate(starts, start=1):
        train_idx = price.index[start_idx - train_len : start_idx]
        test_idx = price.index[start_idx : start_idx + test_len]
        train_scores = {}
        for params, ret in returns_by_param.items():
            train_scores[params] = stats(ret.loc[ret.index.intersection(train_idx)])["sharpe"]
        best_params = max(train_scores, key=lambda key: -np.inf if np.isnan(train_scores[key]) else train_scores[key])
        test_ret = returns_by_param[best_params].loc[returns_by_param[best_params].index.intersection(test_idx)]
        oos_parts.append(test_ret)
        records.append(
            {
                "fold": fold,
                "start": test_idx[0].date(),
                "best": f"{best_params[0]}/{best_params[1]}",
                "train_sharpe": train_scores[best_params],
                "test_sharpe": stats(test_ret)["sharpe"],
                "test_return": stats(test_ret)["total_return"],
            }
        )

    result = pd.DataFrame(records)
    oos = pd.concat(oos_parts).sort_index()
    oos_equity = (1 + oos).cumprod()
    buyhold = (1 + price.pct_change().reindex(oos.index).fillna(0)).cumprod()

    fig = plt.figure(figsize=(13, 7), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.54, 0.46])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    for ax in [ax1, ax2, ax3]:
        set_dark(ax)

    x = np.arange(len(result))
    ax1.bar(x - 0.18, result["train_sharpe"], width=0.36, label="train", color=ACCENT)
    ax1.bar(x + 0.18, result["test_sharpe"], width=0.36, label="test", color=GREEN)
    ax1.axhline(0, color=MUTED, linewidth=1)
    ax1.set_xticks(x, [f"F{row.fold}\n{row.best}" for row in result.itertuples()], fontsize=8)
    ax1.set_title("Walk-Forward Selected Parameters", loc="left", fontsize=16, weight="bold")
    ax1.legend(frameon=True, loc="upper left")

    ax2.plot(oos_equity.index, oos_equity * 100, color=CYAN, linewidth=2.0, label="walk-forward")
    ax2.plot(buyhold.index, buyhold * 100, color=MUTED, linewidth=1.5, label="buy & hold")
    ax2.set_title("Out-of-Sample Equity", loc="left", fontsize=13, weight="bold")
    ax2.set_ylabel("Index")
    ax2.legend(frameon=True, loc="upper left")

    ax3.bar(result["fold"].astype(str), result["test_return"] * 100, color=[GREEN if v >= 0 else RED for v in result["test_return"]])
    ax3.axhline(0, color=MUTED, linewidth=1)
    ax3.set_title("Test Return by Fold", loc="left", fontsize=13, weight="bold")
    ax3.set_ylabel("%")

    fig.tight_layout(pad=1.9)
    save(fig, "advanced-walk-forward-matrix.png")


def plot_monte_carlo(frames: dict[str, pd.DataFrame]) -> None:
    symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    prices = panel(frames, symbols)
    returns = prices.pct_change().dropna()
    weights = pd.Series(1 / len(symbols), index=symbols)
    portfolio_returns = returns @ weights

    rng = np.random.default_rng(42)
    horizon = 252
    paths = 600
    samples = rng.choice(portfolio_returns.to_numpy(), size=(horizon, paths), replace=True)
    simulated = 100 * np.cumprod(1 + samples, axis=0)
    bands = np.percentile(simulated, [5, 25, 50, 75, 95], axis=1)
    realized = 100 * (1 + portfolio_returns.tail(horizon)).cumprod()
    terminal_returns = simulated[-1] / 100 - 1

    fig = plt.figure(figsize=(13, 7), facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.66, 0.34])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for ax in [ax1, ax2]:
        set_dark(ax)
    days = np.arange(horizon)
    ax1.fill_between(days, bands[0], bands[4], color=ACCENT, alpha=0.14, label="5-95%")
    ax1.fill_between(days, bands[1], bands[3], color=ACCENT, alpha=0.24, label="25-75%")
    ax1.plot(days, bands[2], color=CYAN, linewidth=2.2, label="median")
    ax1.plot(days[-len(realized) :], realized.values, color=YELLOW, linewidth=1.8, label="realized trailing year")
    ax1.set_title("Bootstrap Monte Carlo Fan", loc="left", fontsize=16, weight="bold")
    ax1.set_xlabel("Trading days")
    ax1.set_ylabel("Portfolio value")
    ax1.legend(frameon=True, loc="upper left")

    ax2.hist(terminal_returns * 100, bins=34, color=PURPLE, alpha=0.82)
    var_5 = np.percentile(terminal_returns, 5) * 100
    median = np.percentile(terminal_returns, 50) * 100
    ax2.axvline(var_5, color=RED, linewidth=2, label=f"5th pct {var_5:.1f}%")
    ax2.axvline(median, color=CYAN, linewidth=2, label=f"median {median:.1f}%")
    ax2.set_title("One-Year Terminal Return", loc="left", fontsize=13, weight="bold")
    ax2.set_xlabel("%")
    ax2.legend(frameon=True, loc="upper left")
    fig.tight_layout(pad=1.9)
    save(fig, "advanced-monte-carlo-tail-risk.png")


def plot_drawdown_diagnostics(frames: dict[str, pd.DataFrame]) -> None:
    price = frames["SPY"]["price"].dropna()
    sharpe, _, _, best = sma_grid(price)
    ret = strategy_returns(price, best[0], best[1])
    buyhold_ret = price.pct_change().reindex(ret.index).fillna(0.0)
    equity = (1 + ret).cumprod()
    buyhold = (1 + buyhold_ret).cumprod()
    drawdown = equity / equity.cummax() - 1
    rolling_vol = ret.rolling(63).std() * np.sqrt(252)
    fast_ma = price.rolling(best[0]).mean()
    slow_ma = price.rolling(best[1]).mean()
    position = (fast_ma > slow_ma).astype(float).shift(1).reindex(ret.index).fillna(0.0)

    fig = plt.figure(figsize=(13, 7.4), facecolor=BG)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.42, 0.32, 0.26])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])
    for ax in [ax1, ax2, ax3, ax4]:
        set_dark(ax)

    ax1.plot(equity.index, equity * 100, color=CYAN, linewidth=2.0, label=f"SMA {best[0]}/{best[1]}")
    ax1.plot(buyhold.index, buyhold * 100, color=MUTED, linewidth=1.4, label="buy & hold")
    ax1.set_title("Strategy Equity vs Benchmark", loc="left", fontsize=16, weight="bold")
    ax1.legend(frameon=True, loc="upper left")

    ax2.fill_between(drawdown.index, drawdown * 100, 0, color=RED, alpha=0.78)
    ax2.set_title("Underwater Drawdown", loc="left", fontsize=13, weight="bold")
    ax2.set_ylabel("%")

    ax3.plot(rolling_vol.index, rolling_vol * 100, color=ORANGE, linewidth=1.8)
    ax3.set_title("Rolling 63D Volatility", loc="left", fontsize=13, weight="bold")
    ax3.set_ylabel("%")

    ax4.fill_between(position.index, 0, position, color=GREEN, alpha=0.42)
    ax4.set_ylim(-0.05, 1.05)
    ax4.set_yticks([0, 1], ["cash", "long"])
    ax4.set_title("Exposure State", loc="left", fontsize=13, weight="bold")

    fig.tight_layout(pad=1.9)
    save(fig, "advanced-drawdown-diagnostics.png")


def mean_upper_triangle(corr: pd.DataFrame) -> float:
    arr = corr.to_numpy(dtype=float)
    mask = np.triu(np.ones(arr.shape, dtype=bool), k=1)
    return float(np.nanmean(arr[mask]))


def plot_correlation_regime(frames: dict[str, pd.DataFrame]) -> None:
    symbols = ["SPY", "TLT", "GLD", "DBC", "UUP"]
    prices = panel(frames, symbols)
    returns = prices.pct_change().dropna()
    recent_corr = returns.tail(252).corr()

    rolling_rows = []
    for idx in range(63, len(returns)):
        window = returns.iloc[idx - 63 : idx]
        corr = window.corr()
        rolling_rows.append(
            {
                "date": returns.index[idx],
                "avg_corr": mean_upper_triangle(corr),
                "spy_tlt": corr.loc["SPY", "TLT"],
                "spy_gld": corr.loc["SPY", "GLD"],
            }
        )
    rolling = pd.DataFrame(rolling_rows).set_index("date")
    eq = (1 + returns["SPY"]).cumprod()
    eq_dd = eq / eq.cummax() - 1

    fig = plt.figure(figsize=(13, 7.2), facecolor=BG)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    for ax in [ax1, ax2, ax3]:
        set_dark(ax)

    im = ax1.imshow(recent_corr.values, vmin=-1, vmax=1, cmap="RdBu")
    ax1.set_xticks(range(len(symbols)), symbols, rotation=45, ha="right")
    ax1.set_yticks(range(len(symbols)), symbols)
    ax1.set_title("Recent Cross-Asset Correlation", loc="left", fontsize=15, weight="bold")
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            ax1.text(j, i, f"{recent_corr.iloc[i, j]:.2f}", ha="center", va="center", color=TEXT, fontsize=9)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    style_colorbar(cbar)

    ax2.plot(rolling.index, rolling["avg_corr"], color=CYAN, linewidth=1.8, label="avg pairwise")
    ax2.plot(rolling.index, rolling["spy_tlt"], color=ORANGE, linewidth=1.4, label="SPY/TLT")
    ax2.plot(rolling.index, rolling["spy_gld"], color=GREEN, linewidth=1.4, label="SPY/GLD")
    ax2.axhline(0, color=MUTED, linewidth=1)
    ax2.set_title("Rolling 63D Correlation", loc="left", fontsize=13, weight="bold")
    ax2.legend(frameon=True, loc="upper left")

    ax3.fill_between(eq_dd.index, eq_dd * 100, 0, color=RED, alpha=0.72)
    ax3.set_title("SPY Drawdown Context", loc="left", fontsize=13, weight="bold")
    ax3.set_ylabel("%")

    fig.tight_layout(pad=1.9)
    save(fig, "advanced-correlation-regime-map.png")


def rolling_factor_betas(portfolio: pd.Series, factors: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    aligned = pd.concat([portfolio.rename("portfolio"), factors], axis=1).dropna()
    rows = []
    for idx in range(window, len(aligned)):
        chunk = aligned.iloc[idx - window : idx]
        y = chunk["portfolio"].to_numpy()
        x = chunk[factors.columns].to_numpy()
        x = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(x, y, rcond=None)[0][1:]
        rows.append(dict(date=aligned.index[idx], **{col: beta[i] for i, col in enumerate(factors.columns)}))
    return pd.DataFrame(rows).set_index("date")


def plot_factor_exposure(frames: dict[str, pd.DataFrame]) -> None:
    holdings = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    factors = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    holding_prices = panel(frames, holdings)
    factor_prices = panel(frames, factors)
    holding_returns = holding_prices.pct_change().dropna()
    factor_returns = factor_prices.pct_change().dropna()
    weights = pd.Series(1 / len(holdings), index=holdings)
    portfolio_returns = holding_returns @ weights
    betas = rolling_factor_betas(portfolio_returns, factor_returns)
    latest = betas.iloc[-1]
    factor_quarter = factor_returns.tail(63).sum()
    contribution = latest * factor_quarter

    fig = plt.figure(figsize=(13, 7), facecolor=BG)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    for ax in [ax1, ax2, ax3]:
        set_dark(ax)

    colors = [ACCENT, PURPLE, GREEN, ORANGE, YELLOW]
    for factor, color in zip(factors, colors):
        ax1.plot(betas.index, betas[factor], color=color, linewidth=1.5, label=factor)
    ax1.axhline(0, color=MUTED, linewidth=1)
    ax1.set_title("Rolling 126D Factor Betas", loc="left", fontsize=16, weight="bold")
    ax1.legend(frameon=True, ncols=2, loc="upper left")

    ax2.bar(latest.index, latest.values, color=colors)
    ax2.axhline(0, color=MUTED, linewidth=1)
    ax2.set_title("Latest Exposure", loc="left", fontsize=13, weight="bold")

    ax3.bar(contribution.index, contribution.values * 100, color=[GREEN if v >= 0 else RED for v in contribution.values])
    ax3.axhline(0, color=MUTED, linewidth=1)
    ax3.set_title("Quarter Factor Contribution Proxy", loc="left", fontsize=13, weight="bold")
    ax3.set_ylabel("%")

    fig.tight_layout(pad=1.9)
    save(fig, "advanced-factor-exposure-diagnostics.png")


def notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def write_notebooks() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        {
            "file": "61_vectorized_strategy_grid.ipynb",
            "title": "Vectorized Strategy Grid",
            "domain": "/d/equity/pricing input + vectorized research diagnostics",
            "image": "advanced-sma-grid-heatmap.png",
            "summary": "SMA parameter sweep with transaction costs and Sharpe heatmap.",
            "snippet": """raw = qj.eod.get_historical_prices(
    symbol="SPY",
    start_date="2020-01-01",
    end_date="2026-06-06",
)

rows = raw["data"]["value"]
prices = pd.DataFrame(rows).assign(
    date=lambda df: pd.to_datetime(df["date"]),
    price=lambda df: pd.to_numeric(df["adjusted_close"].fillna(df["close"]))
).set_index("date")["price"]

fast_windows = [5, 10, 15, 20, 30, 40, 50]
slow_windows = [60, 80, 100, 125, 150, 200]
# Then evaluate the full SMA cross grid with one-day signal lag and 5 bps turnover cost.""",
        },
        {
            "file": "62_walk_forward_robustness.ipynb",
            "title": "Walk-Forward Robustness",
            "domain": "/d/equity/pricing input + walk-forward validation",
            "image": "advanced-walk-forward-matrix.png",
            "summary": "Train/test parameter selection across rolling folds.",
            "snippet": """raw = qj.eod.get_historical_prices(
    symbol="SPY",
    start_date="2020-01-01",
    end_date="2026-06-06",
)

train_days = 504
test_days = 126
parameter_grid = {
    "fast": [5, 10, 15, 20, 30, 40, 50],
    "slow": [60, 80, 100, 125, 150, 200],
}
# For each fold: choose best train Sharpe, then evaluate the next 126 trading days.""",
        },
        {
            "file": "63_monte_carlo_tail_risk.ipynb",
            "title": "Monte Carlo Tail Risk",
            "domain": "/d/equity/pricing input + portfolio tail-risk simulation",
            "image": "advanced-monte-carlo-tail-risk.png",
            "summary": "Bootstrap fan chart and terminal-return distribution for a mega-cap equity basket.",
            "snippet": """symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
prices = {
    symbol: qj.eod.get_historical_prices(
        symbol=symbol,
        start_date="2020-01-01",
        end_date="2026-06-06",
    )
    for symbol in symbols
}

# Normalize each response, build equal-weight returns, bootstrap 600 one-year paths.""",
        },
        {
            "file": "64_correlation_regime_lab.ipynb",
            "title": "Correlation Regime Lab",
            "domain": "/d/equity/pricing input + cross-asset correlation regime",
            "image": "advanced-correlation-regime-map.png",
            "summary": "Cross-asset heatmap with rolling pairwise correlation and drawdown context.",
            "snippet": """symbols = ["SPY", "TLT", "GLD", "DBC", "UUP"]
prices = {
    symbol: qj.eod.get_historical_prices(
        symbol=symbol,
        start_date="2020-01-01",
        end_date="2026-06-06",
    )
    for symbol in symbols
}

# Build daily returns, recent 252D correlation and rolling 63D pairwise correlation.""",
        },
        {
            "file": "65_drawdown_diagnostics.ipynb",
            "title": "Drawdown Diagnostics",
            "domain": "/d/equity/pricing input + strategy diagnostics",
            "image": "advanced-drawdown-diagnostics.png",
            "summary": "Underwater chart, rolling volatility and exposure state for a selected strategy.",
            "snippet": """raw = qj.eod.get_historical_prices(
    symbol="SPY",
    start_date="2020-01-01",
    end_date="2026-06-06",
)

# Compute SMA 5/125 equity, underwater drawdown, rolling 63D vol and exposure state.""",
        },
        {
            "file": "66_factor_exposure_diagnostics.ipynb",
            "title": "Factor Exposure Diagnostics",
            "domain": "/d/equity/pricing input + rolling factor exposure",
            "image": "advanced-factor-exposure-diagnostics.png",
            "summary": "Rolling factor betas and recent contribution proxy for an equity basket.",
            "snippet": """holdings = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
factors = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

prices = {
    symbol: qj.eod.get_historical_prices(
        symbol=symbol,
        start_date="2020-01-01",
        end_date="2026-06-06",
    )
    for symbol in holdings + factors
}

# Estimate rolling 126D betas with numpy.linalg.lstsq over normalized returns.""",
        },
    ]

    for spec in specs:
        nb = notebook(
            [
                markdown(f"# {spec['title']}\n\n{spec['summary']}\n\nDomain: `{spec['domain']}`"),
                code(
                    """import os
from quantjourney.sdk import QuantJourneyAPI

qj = QuantJourneyAPI(api_key=os.environ["QJ_API_KEY"])
"""
                ),
                code(spec["snippet"]),
                markdown(f"![{spec['title']}](../../outputs/buy_side_advanced/{spec['image']})"),
                markdown(
                    "Generated by `scripts/generate_advanced_buy_side_outputs.py` from live QuantJourney API price data."
                ),
            ]
        )
        (NOTEBOOK_DIR / spec["file"]).write_text(json.dumps(nb, indent=2), encoding="utf-8")


def update_manifest() -> None:
    existing: list[dict[str, Any]] = []
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    existing = [item for item in existing if item.get("group") != "buy_side_advanced"]
    for nb_path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        existing.append(
            {
                "name": nb_path.name,
                "group": "buy_side_advanced",
                "path": str(nb_path.relative_to(ROOT)),
                "output_dir": str(OUTPUT_DIR.relative_to(ROOT)),
                "images": 1,
                "cells": 5,
            }
        )
    MANIFEST.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    api_key = os.environ.get("QJ_API_KEY")
    if not api_key:
        raise SystemExit("Set QJ_API_KEY before running this script.")

    qj = QuantJourneyAPI(api_key=api_key)
    symbols = sorted({"SPY", "QQQ", "IWM", "TLT", "GLD", "DBC", "UUP", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"})
    frames = fetch_price_panel(qj, symbols)

    grid_info = plot_sma_grid(frames)
    plot_walk_forward(frames)
    plot_monte_carlo(frames)
    plot_drawdown_diagnostics(frames)
    plot_correlation_regime(frames)
    plot_factor_exposure(frames)
    write_notebooks()
    update_manifest()

    print(f"Best SMA grid parameters: {grid_info['best_fast']}/{grid_info['best_slow']}")
    print(f"Wrote advanced buy-side outputs to {OUTPUT_DIR}")
    print(f"Wrote notebooks to {NOTEBOOK_DIR}")


if __name__ == "__main__":
    main()
