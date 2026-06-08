"""Generate landing-page chart outputs from live QuantJourney API calls.

The script expects QJ_API_KEY in the environment and writes PNG files into
outputs/landing. It uses the current QuantJourney response shape:
{"data": {"value": ...}, "meta": ..., "warnings": ...}.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantjourney.sdk import QuantJourneyAPI


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "landing"
END_DATE = os.environ.get("QJ_LANDING_END_DATE") or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

BG = "#020817"
PANEL = "#061641"
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

plt.rcParams.update(
    {
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 11,
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
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    numeric_cols = ["open", "high", "low", "close", "adjusted_close", "volume"]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["price"] = df["adjusted_close"].fillna(df["close"]) if "adjusted_close" in df else df["close"]
    return df.set_index("date")


def series_frame(raw: Any) -> pd.DataFrame:
    rows = value(raw)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No series data returned")
    df["date"] = pd.to_datetime(df["date"])
    value_cols = [c for c in df.columns if c not in {"date", "realtime_start", "realtime_end"}]
    if not value_cols:
        raise RuntimeError("Series payload has no value column")
    col = value_cols[0]
    df["value"] = pd.to_numeric(df[col], errors="coerce")
    return df[["date", "value"]].dropna().sort_values("date")


def set_dark(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.7, linewidth=0.8)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.title.set_color(TEXT)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.tick_params(colors=MUTED)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_color(TEXT)
    fig.savefig(OUTPUT_DIR / name, facecolor=BG, edgecolor=BG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def market_data(qj: QuantJourneyAPI) -> None:
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    prices = {}
    for symbol in symbols:
        df = price_frame(qj, symbol, "2024-01-01", "2024-12-31")
        prices[symbol] = df["price"]
    combined = pd.DataFrame(prices).dropna()
    normalized = combined / combined.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=BG)
    colors = [ACCENT, RED, GREEN, PURPLE, ORANGE]
    for symbol, color in zip(symbols, colors):
        ax.plot(normalized.index, normalized[symbol], label=symbol, linewidth=2.1, color=color)
    ax.axhline(100, color=MUTED, linestyle="--", linewidth=1.1, alpha=0.8)
    ax.set_title("Adjusted Performance Comparison (2024)", loc="left", fontsize=17, weight="bold")
    ax.set_ylabel("Normalized price, base = 100")
    ax.legend(frameon=False, labelcolor=TEXT, ncols=5, loc="upper left", bbox_to_anchor=(0, -0.13))
    set_dark(ax)
    save(fig, "recipe-market-data-real.png")


def macro_dashboard(qj: QuantJourneyAPI) -> None:
    gdp = series_frame(qj.fred.get_gdp())
    cpi = series_frame(qj.fred.get_cpi())
    unemp = series_frame(qj.fred.get_unemployment_rate())
    fed = series_frame(qj.fred.get_effective_federal_funds_rate())

    gdp["yoy"] = gdp["value"].pct_change(4) * 100
    cpi["yoy"] = cpi["value"].pct_change(12) * 100
    cutoff = pd.Timestamp("2010-01-01")

    fig, axs = plt.subplots(2, 2, figsize=(12, 6.5), facecolor=BG)
    charts = [
        (axs[0, 0], gdp[gdp["date"] >= cutoff], "GDP Growth YoY", "yoy", ACCENT, "bar"),
        (axs[0, 1], cpi[cpi["date"] >= cutoff], "Inflation CPI YoY", "yoy", RED, "line"),
        (axs[1, 0], unemp[unemp["date"] >= cutoff], "Unemployment Rate", "value", GREEN, "line"),
        (axs[1, 1], fed[fed["date"] >= cutoff], "Fed Funds Rate", "value", PURPLE, "line"),
    ]
    for ax, df, title, col, color, kind in charts:
        set_dark(ax)
        ax.set_title(title, loc="left", fontsize=13, weight="bold")
        if kind == "bar":
            ax.bar(df["date"], df[col], width=70, color=color)
        else:
            ax.plot(df["date"], df[col], color=color, linewidth=2)
            ax.fill_between(df["date"], df[col], color=color, alpha=0.22)
    fig.suptitle("US Macro Dashboard", x=0.03, y=0.99, ha="left", color=TEXT, fontsize=17, weight="bold")
    fig.tight_layout(pad=2.0)
    save(fig, "recipe-macro-dashboard-real.png")


def peer_valuation(qj: QuantJourneyAPI) -> None:
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    rows = []
    for symbol in symbols:
        ratios = value(qj.fmp.get_financial_ratios_ttm(symbol=symbol))
        rows.append(
            {
                "symbol": symbol,
                "P/E": float(ratios.get("peRatioTTM") or ratios.get("priceToEarningsRatioTTM") or np.nan),
                "P/B": float(ratios.get("priceToBookTTM") or ratios.get("priceToBookRatioTTM") or np.nan),
                "Gross margin": float(ratios.get("grossProfitMarginTTM") or np.nan) * 100,
                "FCF yield": 100 / float(ratios.get("priceToFreeCashFlowRatioTTM") or np.nan),
            }
        )
    df = pd.DataFrame(rows).set_index("symbol")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6.5), facecolor=BG)
    palette = [ACCENT, RED, GREEN, PURPLE, ORANGE]
    for ax, metric in zip(axs, ["P/E", "Gross margin"]):
        set_dark(ax)
        ax.bar(df.index, df[metric], color=palette)
        ax.set_title(metric, loc="left", fontsize=14, weight="bold")
        ax.axhline(df[metric].mean(), color=MUTED, linestyle="--", linewidth=1)
    fig.suptitle("Peer Valuation and Quality", x=0.03, y=0.96, ha="left", color=TEXT, fontsize=17, weight="bold")
    fig.tight_layout(pad=2.0)
    save(fig, "recipe-peer-valuation-real.png")


def portfolio_risk(qj: QuantJourneyAPI) -> None:
    weights = {"AAPL": 0.20, "MSFT": 0.20, "NVDA": 0.20, "GOOGL": 0.20, "AMZN": 0.20}
    prices = {s: price_frame(qj, s, "2024-01-01", "2025-12-31")["price"] for s in weights}
    price_df = pd.DataFrame(prices).dropna()
    returns = price_df.pct_change().dropna()
    w = pd.Series(weights).reindex(returns.columns)
    portfolio_returns = returns @ w
    cumulative = (1 + portfolio_returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    corr = returns.corr()
    cov = returns.cov() * 252
    port_vol = float(np.sqrt(w.values @ cov.values @ w.values))
    risk_contrib = w.values * (cov.values @ w.values) / (port_vol**2)

    fig = plt.figure(figsize=(12, 6.5), facecolor=BG)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    for ax in [ax1, ax2, ax3]:
        set_dark(ax)
    im = ax1.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu")
    ax1.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax1.set_yticks(range(len(corr.index)), corr.index)
    ax1.set_title("Correlation Matrix", loc="left", fontsize=14, weight="bold")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax2.plot(cumulative.index, cumulative.values * 100, color=CYAN, linewidth=2)
    ax2.set_title("Portfolio Value", loc="left", fontsize=13, weight="bold")
    ax3.fill_between(drawdown.index, drawdown.values * 100, color=RED, alpha=0.75)
    ax3.set_title("Drawdown", loc="left", fontsize=13, weight="bold")
    ax3.set_ylabel("%")
    fig.suptitle("Portfolio Risk Snapshot", x=0.03, y=0.98, ha="left", color=TEXT, fontsize=17, weight="bold")
    fig.tight_layout(pad=2.0)
    save(fig, "recipe-portfolio-risk-real.png")

    fig2, ax = plt.subplots(figsize=(12, 6.5), facecolor=BG)
    set_dark(ax)
    ax.bar(returns.columns, risk_contrib * 100, color=[ACCENT, RED, GREEN, PURPLE, ORANGE])
    ax.set_title("Risk Contribution", loc="left", fontsize=17, weight="bold")
    ax.set_ylabel("% of portfolio variance")
    save(fig2, "recipe-portfolio-risk-contribution-real.png")


def vix_regime(qj: QuantJourneyAPI) -> None:
    raw = value(qj.cboe.get_vix_data(start_date="2020-01-01", end_date=END_DATE))
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna().sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=BG)
    set_dark(ax)
    ax.plot(df["date"], df["close"], color=ORANGE, linewidth=1.8)
    ax.fill_between(df["date"], df["close"], color=ORANGE, alpha=0.28)
    for y, label, color in [(20, "low fear", GREEN), (30, "high fear", ORANGE), (40, "extreme fear", RED)]:
        ax.axhline(y, color=color, linestyle="--", linewidth=1)
        ax.text(df["date"].iloc[int(len(df) * 0.77)], y + 0.8, label, color=color, fontsize=10)
    ax.set_title("VIX Regime", loc="left", fontsize=17, weight="bold")
    ax.set_ylabel("VIX close")
    save(fig, "recipe-vix-regime-real.png")


def risk_parity(qj: QuantJourneyAPI) -> None:
    symbols = ["SPY", "TLT", "GLD", "DBC", "UUP"]
    prices = {s: price_frame(qj, s, "2021-01-01", "2025-12-31")["price"] for s in symbols}
    price_df = pd.DataFrame(prices).dropna()
    returns = price_df.pct_change().dropna()
    vol = returns.std() * np.sqrt(252)
    inv = 1 / vol
    weights = inv / inv.sum()
    cov = returns.cov() * 252
    port_vol = float(np.sqrt(weights.values @ cov.values @ weights.values))
    risk_contrib = weights.values * (cov.values @ weights.values) / (port_vol**2)
    cumulative = (1 + returns @ weights).cumprod()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6.5), facecolor=BG)
    for ax in axs:
        set_dark(ax)
    colors = [ACCENT, RED, GREEN, PURPLE, ORANGE]
    axs[0].bar(symbols, weights.values * 100, color=colors)
    axs[0].set_title("Inverse-Volatility Weights", loc="left", fontsize=14, weight="bold")
    axs[0].set_ylabel("% weight")
    axs[1].bar(symbols, risk_contrib * 100, color=colors)
    axs[1].set_title("Risk Contribution", loc="left", fontsize=14, weight="bold")
    axs[1].set_ylabel("% risk")
    fig.suptitle("Risk Parity Allocation", x=0.03, y=0.96, ha="left", color=TEXT, fontsize=17, weight="bold")
    fig.tight_layout(pad=2.0)
    save(fig, "recipe-risk-parity-real.png")

    fig2, ax = plt.subplots(figsize=(12, 6.5), facecolor=BG)
    set_dark(ax)
    ax.plot(cumulative.index, (cumulative - 1) * 100, color=CYAN, linewidth=2)
    ax.set_title("Risk Parity Cumulative Performance", loc="left", fontsize=17, weight="bold")
    ax.set_ylabel("Return (%)")
    save(fig2, "recipe-risk-parity-performance-real.png")


def main() -> None:
    api_key = os.environ.get("QJ_API_KEY")
    if not api_key:
        raise SystemExit("Set QJ_API_KEY before running this script.")
    qj = QuantJourneyAPI(api_key=api_key)
    market_data(qj)
    macro_dashboard(qj)
    peer_valuation(qj)
    portfolio_risk(qj)
    vix_regime(qj)
    risk_parity(qj)
    print(f"Wrote landing outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
