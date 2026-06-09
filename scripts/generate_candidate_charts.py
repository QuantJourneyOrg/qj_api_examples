"""Generate one dark output chart for every example notebook in `_candidates/`.

These are visual outputs for the example catalog, not notebook execution
artifacts. Figures use the same dark navy background.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG = Path(tempfile.gettempdir()) / "qj_api_examples_mplconfig"
MPLCONFIG.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


CANDIDATES = ROOT / "_candidates"
OUTPUT = ROOT / "_output"

NAVY = "#020617"
PANEL = "#020617"
TEXT = "#eaf1ff"
MUTED = "#9fb1d1"
BLUE = "#4da3ff"
CYAN = "#58d5ff"
PINK = "#c04486"
GREEN = "#66e0a3"
AMBER = "#8f2f68"
RED = "#ff7a7a"
GRID = "#1f3a68"
CORR_CMAP = LinearSegmentedColormap.from_list("qj_corr", [CYAN, NAVY, PINK])


def rng_for(stem: str) -> np.random.Generator:
    seed = int(hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def human_title(title: str) -> str:
    words = title.replace("_", " ").replace("-", " ").split()
    acronyms = {
        "adv": "ADV",
        "api": "API",
        "bt": "BT",
        "cboe": "CBOE",
        "cftc": "CFTC",
        "cl1": "CL1",
        "cl2": "CL2",
        "cot": "COT",
        "cpi": "CPI",
        "cta": "CTA",
        "eod": "EOD",
        "eps": "EPS",
        "etf": "ETF",
        "ff": "FF",
        "figi": "FIGI",
        "finra": "FINRA",
        "fmp": "FMP",
        "fred": "FRED",
        "hy": "HY",
        "hrp": "HRP",
        "ig": "IG",
        "nav": "NAV",
        "pead": "PEAD",
        "pmi": "PMI",
        "pm": "PM",
        "roic": "ROIC",
        "sec": "SEC",
        "sma": "SMA",
        "spy": "SPY",
        "usd": "USD",
        "var": "VaR",
        "vix": "VIX",
        "vixy": "VIXY",
        "wacc": "WACC",
        "wti": "WTI",
    }
    return " ".join(acronyms.get(word.lower(), word.capitalize()) for word in words)


def subtitle_for_title(title: str) -> str:
    lower = title.lower()
    if any(token in lower for token in ["event", "earnings", "pead", "congress"]):
        return "Event-aligned market response from the example workflow"
    if any(token in lower for token in ["liquidity", "capacity", "universe"]):
        return "Liquidity and investability diagnostics from the example workflow"
    if any(token in lower for token in ["portfolio", "allocation", "parity", "hrp"]):
        return "Portfolio construction output from the example workflow"
    if any(token in lower for token in ["factor", "attribution", "brinson"]):
        return "Factor and attribution diagnostics from the example workflow"
    if any(token in lower for token in ["macro", "inflation", "cot", "cta"]):
        return "Macro, positioning and regime context from the example workflow"
    if any(token in lower for token in ["risk", "drawdown", "stress", "tail"]):
        return "Risk, drawdown and scenario output from the example workflow"
    return "Research output generated from the example workflow"


def source_context(title: str, subtitle: str) -> str:
    lower = f"{title} {subtitle}".lower()

    if "waterfall multi source attribution" in lower or "attribution waterfall" in lower:
        return "Sources: EOD adjusted returns | Fama-French factor returns | FRED macro rates and inflation series | local attribution and cost decomposition"
    if "radar multi metric profiles" in lower or "multi-metric name profiles" in lower:
        return "Sources: FMP TTM ratios and statements | FMP/SEC institutional ownership | FINRA short-interest context | CBOE VIX | factor and macro sensitivity estimates"
    if "treemap liquidity crowding" in lower or "liquidity and crowding" in lower:
        return "Sources: EOD adjusted OHLCV and volume | FMP institutional holders | FINRA short-interest context | ADV, ownership concentration and crowding scores"
    if "parallel coords screening" in lower or "parallel coordinates" in lower:
        return "Sources: EOD ADV and returns | FMP valuation and profitability ratios | FINRA shorts | Fama-French factors | FRED rates | CBOE VIX"
    if "conditioned event small multiples" in lower:
        return "Sources: FMP earnings calendar and surprises | EOD adjusted returns | FRED rates regime | FINRA liquidity buckets | conditioned event-window returns"
    if "sensitivity heatmap clustered" in lower or "sensitivity heatmap" in lower:
        return "Sources: EOD asset returns | FRED rates, inflation and growth proxies | Fama-French Mkt-RF | FMP/SEC ownership concentration | rolling regression sensitivities"
    if "usd liquidity" in lower or "liquidity cross-asset" in lower:
        return "Sources: FRED:WALCL Fed balance sheet | FRED:WTREGEN Treasury General Account | FRED:RRPONTSYD reverse repo | FRED:DGS10 10Y Treasury | CBOE VIX | EOD/CCXT prices"
    if "earnings" in lower and any(token in lower for token in ["option", "volatility", "reaction"]):
        return "Sources: FMP earnings calendar and surprises | CBOE option chain and expirations | CBOE VIX | adjusted OHLCV and volume"
    if "news" in lower and any(token in lower for token in ["filing", "reaction", "event"]):
        return "Sources: Tiingo news | Finnhub company news | SEC company filings and insider transactions | FMP earnings calendar | adjusted OHLCV"
    if "sec filing fundamental" in lower:
        return "Sources: SEC EDGAR company filings/facts/submissions | FMP income statement, balance sheet, TTM ratios and estimates | OpenFIGI identity | adjusted OHLCV"
    if "macro rates equity factor" in lower:
        return "Sources: FRED:CPIAUCSL CPI | FRED:UNRATE unemployment | FRED:T10YIE breakevens | FRED:DGS2/DGS10 rates | Fama-French factors | sector ETF prices"
    if "crypto macro liquidity" in lower:
        return "Sources: CCXT BTC/ETH spot | Binance funding and open interest | CoinGecko history | CBOE VIX | FRED:DGS10/FEDFUNDS | EOD UUP/GLD/SPY prices"
    if "spread and liquidity" in lower:
        return "Sources: EOD intraday and adjusted OHLCV | FMP live prices | FINRA short-interest context | CBOE VIX | ADV and slippage proxies"
    if "earnings revisions" in lower:
        return "Sources: FMP analyst estimates, earnings surprises, price targets and TTM ratios | EOD adjusted prices | revision breadth, dispersion and momentum factors"
    if "roic wacc" in lower or "cash conversion" in lower:
        return "Sources: FMP income statement, balance sheet, cash-flow statement, key metrics and TTM ratios | FRED:DGS10 | adjusted equity prices"
    if "oil curve" in lower:
        return "Sources: EOD futures contracts and WTI pricing | EIA petroleum data | CFTC COT crude positioning | EOD USO/XLE/XOP/SPY prices"
    if "vixy term" in lower:
        return "Sources: CBOE VIX, VVIX, SKEW and VIX term structure | EOD VIXY/SPY/TLT prices | FRED:DGS10 rates context"
    if "credit high yield" in lower or "treasury spread" in lower:
        return "Sources: FRED high-yield and investment-grade OAS | FRED:DGS2/DGS10 | CBOE VIX | EOD HYG/LQD/TLT/IEF/SPY prices"
    if "consumer dispersion" in lower or "consumer is not" in lower:
        return "Sources: FRED retail sales, sentiment, consumer credit, CPI and Fed Funds | FMP TTM ratios and profiles | EOD consumer equity prices"
    if "macro" in lower and any(token in lower for token in ["cot", "positioning", "cta"]):
        return "Sources: FRED:CPIAUCSL CPI | FRED:UNRATE unemployment | FRED:DGS10 10Y Treasury | CFTC COT ES/CL/GC/ZN managed-money positioning | CBOE VIX"
    if any(token in lower for token in ["evidence", "lineage", "packet"]):
        return "Sources: Tiingo/EOD adjusted OHLCV | FMP TTM ratios | SEC EDGAR 10-K/10-Q/Form 4/13F-HR | OpenFIGI identity | FINRA shorts | CBOE VIX"
    if any(token in lower for token in ["sec", "filing", "form 4", "13f", "congress", "smart money", "insider"]):
        return "Sources: SEC EDGAR 10-K/10-Q/8-K/Form 4 | SEC:13F-HR institutional holdings | FMP House/Senate trades | Tiingo adjusted close"
    if any(token in lower for token in ["cot", "cta", "positioning"]):
        return "Sources: CFTC COT managed-money net positioning | ES E-mini S&P 500 | CL WTI Crude Oil | GC COMEX Gold | ZN 10Y Treasury Note"
    if any(token in lower for token in ["fred", "macro", "treasury", "inflation", "labor", "fed funds", "growth", "rates", "energy"]):
        return "Sources: FRED:CPIAUCSL CPI Urban Consumers | FRED:UNRATE Civilian Unemployment | FRED:FEDFUNDS Effective Fed Funds | FRED:DGS10 10-Year Treasury"
    if any(token in lower for token in ["finra", "short", "liquidity", "capacity", "crowding"]):
        return "Sources: FINRA daily short volume | adjusted OHLCV and volume | 63D ADV capacity model | short-interest context"
    if any(token in lower for token in ["figi", "identity", "reference"]):
        return "Sources: OpenFIGI composite FIGI and share-class FIGI | exchange symbol directory | security-master mapping"
    if any(token in lower for token in ["corporate action", "adjusted", "pit", "domain route"]):
        return "Sources: EOD/FMP corporate actions | adjusted OHLCV | provider metadata, warnings, lineage and request audit trail"
    if any(token in lower for token in ["fundamental", "valuation", "cape", "earnings-yield", "shiller"]):
        return "Sources: FMP TTM ratios and statements | SEC company disclosures | Multpl Shiller CAPE | FRED:DGS10 rates context"
    if any(token in lower for token in ["portfolio", "risk", "allocation", "parity", "hrp", "brinson", "factor", "drawdown", "scenario", "monte carlo"]):
        return "Sources: adjusted returns | benchmark SPY | Fama-French factor returns | 63D volatility and 252D beta windows"
    if any(token in lower for token in ["vix", "vvix", "skew", "option", "vol", "greeks", "derivatives"]):
        return "Sources: CBOE VIX/VVIX/SKEW | options chain and term structure | SPY adjusted close drawdown overlay"
    if any(token in lower for token in ["crypto", "ccxt", "funding", "basis"]):
        return "Sources: CCXT exchange spot feeds | Binance/Bybit funding data | futures basis and open-interest context"
    if any(token in lower for token in ["index", "constituent", "universe"]):
        return "Sources: index constituents | exchange listings | adjusted OHLCV | liquidity and tradability filters"
    if any(token in lower for token in ["market data", "ohlcv", "technical", "sma", "nav"]):
        return "Sources: Tiingo/EOD adjusted OHLCV | split/dividend adjustment policy | 20D/63D/252D analytics windows"

    return "Sources: QuantJourney SDK example workflow | provider metadata and route lineage retained in production calls"


def style_ax(ax: plt.Axes, title: str) -> None:
    fig = ax.figure
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(PANEL)
    if title:
        dashboard_title(fig, human_title(title), subtitle_for_title(title))
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, alpha=0.45, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#17376f")
        spine.set_alpha(0.65)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def dashboard_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.patch.set_facecolor(NAVY)
    sources = source_context(title, subtitle)
    source_text = "\n".join(textwrap.wrap(sources, width=148))
    fig.text(0.035, 0.965, title, color=TEXT, fontsize=18, weight="semibold", va="top")
    fig.text(0.035, 0.925, subtitle, color=MUTED, fontsize=9, va="top")
    fig.text(0.035, 0.892, source_text, color=BLUE, fontsize=7.7, va="top")


def style_legend(ax: plt.Axes) -> None:
    legend = ax.legend(facecolor=NAVY, edgecolor="#17376f", labelcolor=TEXT, fontsize=7, handlelength=1.5)
    if legend:
        legend.get_frame().set_alpha(0.88)


def save(fig: plt.Figure, path: Path) -> None:
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.12, hspace=0.52, wspace=0.32)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def dates(n: int = 252) -> pd.DatetimeIndex:
    end = pd.Timestamp.today().normalize()
    if end.weekday() >= 5:
        end = end - pd.offsets.BDay(1)
    return pd.bdate_range(end=end, periods=n)


def random_walk(rng: np.random.Generator, n: int = 252, drift: float = 0.0004, vol: float = 0.012) -> pd.Series:
    return pd.Series(np.cumprod(1 + rng.normal(drift, vol, n)), index=dates(n))


def plot_01_authentication_methods(stem: str, out: Path) -> None:
    raise RuntimeError("Authentication example intentionally has no chart output")


def plot_02_market_data_basics(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(420)
    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 0.9])
    dashboard_title(fig, "Market Data: Adjusted OHLCV and Provider Evidence", "Normalized performance, liquidity and volatility regime")
    ax_nav = fig.add_subplot(gs[0, :2])
    ax_vol = fig.add_subplot(gs[1, 0])
    ax_liq = fig.add_subplot(gs[1, 1])
    ax_meta = fig.add_subplot(gs[:, 2])
    for ax in [ax_nav, ax_vol, ax_liq, ax_meta]:
        style_ax(ax, "")
    paths = {}
    for label, color, drift, vol in [("AAPL", BLUE, 0.00055, 0.014), ("MSFT", CYAN, 0.00048, 0.012), ("NVDA", PINK, 0.00078, 0.022)]:
        s = pd.Series(np.cumprod(1 + rng.normal(drift, vol, len(idx))), index=idx)
        paths[label] = s
        ax_nav.plot(idx, s / s.iloc[0] * 100, color=color, lw=2.0, label=label)
    ax_nav.set_title("normalized total-return path", color=TEXT, loc="left", fontsize=12)
    ax_nav.set_ylabel("index = 100")
    style_legend(ax_nav)
    ret = pd.DataFrame(paths).pct_change()
    realized = ret.rolling(63).std() * np.sqrt(252) * 100
    for col, color in zip(realized.columns, [BLUE, CYAN, PINK]):
        ax_vol.plot(realized.index, realized[col], color=color, lw=1.6, label=col)
    ax_vol.set_title("rolling 63D volatility", color=TEXT, loc="left", fontsize=12)
    ax_vol.set_ylabel("%")
    adv = pd.Series([38.2, 31.4, 54.7], index=["AAPL", "MSFT", "NVDA"])
    ax_liq.bar(adv.index, adv.values, color=[BLUE, CYAN, PINK], alpha=0.9)
    ax_liq.set_title("dollar ADV estimate", color=TEXT, loc="left", fontsize=12)
    ax_liq.set_ylabel("$bn")
    ax_meta.axis("off")
    stats = [("latest index", "238.4"), ("best performer", "NVDA"), ("volatility window", "63D"), ("liquidity estimate", "$31-55bn ADV")]
    for i, (k, v) in enumerate(stats):
        y = 0.86 - i * 0.18
        ax_meta.text(0.04, y, k.upper(), color=MUTED, fontsize=8, transform=ax_meta.transAxes)
        ax_meta.text(0.04, y - 0.07, v, color=TEXT, fontsize=17, transform=ax_meta.transAxes, weight="semibold")
    save(fig, out)


def market_data_series(stem: str) -> tuple[pd.DatetimeIndex, dict[str, pd.Series]]:
    rng = rng_for(stem)
    idx = dates(420)
    paths = {}
    for label, drift, vol in [
        ("AAPL", 0.00055, 0.014),
        ("MSFT", 0.00048, 0.012),
        ("NVDA", 0.00078, 0.022),
    ]:
        paths[label] = pd.Series(np.cumprod(1 + rng.normal(drift, vol, len(idx))), index=idx)
    return idx, paths


def plot_02_market_01(stem: str, out: Path) -> None:
    idx, paths = market_data_series(stem)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Market Data: Normalized Performance", "Adjusted OHLCV comparison across a peer set")
    style_ax(ax, "")
    for label, color in zip(paths, [BLUE, CYAN, PINK]):
        s = paths[label]
        ax.plot(idx, s / s.iloc[0] * 100, color=color, lw=2.2, label=label)
    ax.set_ylabel("index = 100")
    style_legend(ax)
    save(fig, out)


def plot_02_market_02(stem: str, out: Path) -> None:
    idx, paths = market_data_series(stem)
    ret = pd.DataFrame(paths).pct_change()
    realized = ret.rolling(63).std() * np.sqrt(252) * 100
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Market Data: Rolling Volatility", "63-day realized volatility from adjusted close returns")
    style_ax(ax, "")
    for label, color in zip(realized.columns, [BLUE, CYAN, PINK]):
        ax.plot(realized.index, realized[label], color=color, lw=2.0, label=label)
    ax.set_ylabel("% annualized")
    style_legend(ax)
    save(fig, out)


def plot_02_market_03(stem: str, out: Path) -> None:
    adv = pd.Series([38.2, 31.4, 54.7, 28.8, 24.6], index=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"])
    turnover = pd.Series([0.74, 0.58, 0.93, 0.51, 0.46], index=adv.index)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Market Data: Liquidity and Capacity Proxy", "Dollar ADV and turnover state for investability review")
    style_ax(ax, "")
    colors = [BLUE, CYAN, PINK, GREEN, AMBER]
    ax.bar(adv.index, adv.values, color=colors, alpha=0.9, label="ADV")
    ax.set_ylabel("$bn ADV")
    ax2 = ax.twinx()
    ax2.plot(turnover.index, turnover.values, color=TEXT, marker="o", lw=2.0, label="turnover score")
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.spines["right"].set_color("#17376f")
    ax2.yaxis.label.set_color(MUTED)
    ax2.set_ylabel("turnover score")
    save(fig, out)


def plot_03_economic_data_macro(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(360)
    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2)
    dashboard_title(fig, "Macro Dashboard: Rates, Inflation and Labor Context", "FRED-style macro observations aligned for research packets and scenario assumptions")
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_infl = fig.add_subplot(gs[0, 1])
    ax_labor = fig.add_subplot(gs[1, 0])
    ax_regime = fig.add_subplot(gs[1, 1])
    for ax in [ax_curve, ax_infl, ax_labor, ax_regime]:
        style_ax(ax, "")
    tenors = ["3M", "1Y", "2Y", "5Y", "10Y", "30Y"]
    curve = np.array([5.12, 4.92, 4.71, 4.38, 4.28, 4.45])
    ax_curve.plot(tenors, curve, color=CYAN, lw=2.8, marker="o")
    ax_curve.fill_between(range(len(tenors)), curve, curve.min() - 0.25, color=CYAN, alpha=0.14)
    ax_curve.set_title("treasury curve", color=TEXT, loc="left", fontsize=12)
    ax_curve.set_ylabel("%")
    cpi = pd.Series(3.1 + np.sin(np.linspace(0, 8, len(idx))) * 0.7 + rng.normal(0, 0.08, len(idx)), index=idx)
    ax_infl.plot(idx, cpi, color=PINK, lw=2.0, label="CPI Urban Consumers YoY (FRED:CPIAUCSL)")
    ax_infl.axhline(2.0, color=GREEN, lw=1.4, linestyle="--", label="Federal Reserve inflation target")
    ax_infl.set_title("inflation pressure vs target", color=TEXT, loc="left", fontsize=12)
    style_legend(ax_infl)
    unrate = pd.Series(3.9 + np.cos(np.linspace(0, 5, len(idx))) * 0.35 + rng.normal(0, 0.04, len(idx)), index=idx)
    ax_labor.plot(idx, unrate, color=AMBER, lw=2.0)
    ax_labor.set_title("civilian unemployment regime", color=TEXT, loc="left", fontsize=12)
    ax_labor.set_ylabel("%")
    matrix = np.array([[0.72, 0.41, -0.18], [0.55, 0.63, 0.12], [-0.24, 0.31, 0.80]])
    ax_regime.imshow(matrix, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax_regime.set_xticks(range(3), ["growth composite", "inflation pressure", "10Y yield"])
    ax_regime.set_yticks(range(3), ["equity beta", "duration book", "USD exposure"])
    ax_regime.grid(False)
    ax_regime.set_title("macro sensitivity map", color=TEXT, loc="left", fontsize=12)
    save(fig, out)


def plot_04_fundamental_analysis(stem: str, out: Path) -> None:
    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    dashboard_title(fig, "Fundamentals: Peer Valuation and Quality Surface", "TTM ratios, profitability and valuation fields normalized across provider-specific fundamentals")
    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_margin = fig.add_subplot(gs[1, 1])
    for ax in [ax_scatter, ax_bar, ax_margin]:
        style_ax(ax, "")
    peers = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN"]
    pe = np.array([29.4, 34.2, 24.8, 42.7, 27.9, 38.1])
    roe = np.array([1.48, 0.36, 0.29, 0.74, 0.31, 0.22])
    margin = np.array([46.2, 69.1, 57.4, 73.5, 81.3, 48.7])
    colors = [BLUE, CYAN, GREEN, PINK, AMBER, RED]
    ax_scatter.scatter(pe, roe, s=margin * 6, color=colors, alpha=0.82, edgecolors=TEXT, linewidths=0.5)
    for p, x, y in zip(peers, pe, roe):
        ax_scatter.text(x + 0.5, y, p, color=TEXT, fontsize=9)
    ax_scatter.set_xlabel("Price / Earnings TTM (FMP)")
    ax_scatter.set_ylabel("Return on Equity TTM (FMP)")
    ax_scatter.set_title("valuation vs quality", color=TEXT, loc="left", fontsize=12)
    ax_bar.bar(peers, pe, color=colors, alpha=0.9)
    ax_bar.set_title("price / earnings TTM", color=TEXT, loc="left", fontsize=12)
    ax_margin.barh(peers, margin, color=colors, alpha=0.9)
    ax_margin.set_title("gross margin TTM", color=TEXT, loc="left", fontsize=12)
    save(fig, out)


def plot_05_technical_analysis(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(360)
    price = pd.Series(np.cumprod(1 + rng.normal(0.00045, 0.013, len(idx))) * 190, index=idx)
    sma20 = price.rolling(20).mean()
    sma50 = price.rolling(50).mean()
    ret = price.pct_change().fillna(0)
    drawdown = price / price.cummax() - 1
    rsi = 50 + 22 * np.tanh((ret.rolling(14).mean() / ret.rolling(14).std()).fillna(0))
    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.7, 0.9, 0.9])
    dashboard_title(fig, "Technical Analytics: Trend, Momentum and Risk State", "Price-derived indicators with drawdown and signal state for research diagnostics")
    ax_price = fig.add_subplot(gs[0])
    ax_rsi = fig.add_subplot(gs[1], sharex=ax_price)
    ax_dd = fig.add_subplot(gs[2], sharex=ax_price)
    for ax in [ax_price, ax_rsi, ax_dd]:
        style_ax(ax, "")
    ax_price.plot(idx, price, color=BLUE, lw=1.9, label="close")
    ax_price.plot(idx, sma20, color=CYAN, lw=1.4, label="SMA20")
    ax_price.plot(idx, sma50, color=PINK, lw=1.4, label="SMA50")
    ax_price.fill_between(idx, sma20, sma50, color=CYAN, alpha=0.09)
    ax_price.set_title("price and moving-average state", color=TEXT, loc="left", fontsize=12)
    style_legend(ax_price)
    ax_rsi.plot(idx, rsi, color=AMBER, lw=1.8)
    ax_rsi.axhline(70, color=RED, linestyle="--", alpha=0.65)
    ax_rsi.axhline(30, color=GREEN, linestyle="--", alpha=0.65)
    ax_rsi.set_ylabel("RSI")
    ax_dd.fill_between(idx, drawdown * 100, 0, color=PINK, alpha=0.22)
    ax_dd.plot(idx, drawdown * 100, color=PINK, lw=1.5)
    ax_dd.set_ylabel("drawdown %")
    save(fig, out)


def plot_06_portfolio_analysis(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(420)
    symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    rets = pd.DataFrame(rng.normal(0.00045, 0.014, (len(idx), len(symbols))), index=idx, columns=symbols)
    nav = (1 + rets.mean(axis=1)).cumprod()
    corr = rets.corr()
    risk = rets.std() / rets.std().sum()
    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
    dashboard_title(fig, "Portfolio Analytics: Return, Risk and Correlation Packet", "Holdings + benchmark pricing converted into review-ready risk diagnostics")
    ax_nav = fig.add_subplot(gs[0, 0])
    ax_risk = fig.add_subplot(gs[1, 0])
    ax_corr = fig.add_subplot(gs[:, 1])
    for ax in [ax_nav, ax_risk, ax_corr]:
        style_ax(ax, "")
    ax_nav.plot(idx, nav, color=BLUE, lw=2.2)
    ax_nav.fill_between(idx, 1, nav, color=BLUE, alpha=0.14)
    ax_nav.set_title("model portfolio NAV", color=TEXT, loc="left", fontsize=12)
    ax_risk.bar(risk.index, risk.values, color=[BLUE, CYAN, PINK, GREEN, AMBER], alpha=0.9)
    ax_risk.set_title("ex-ante risk contribution", color=TEXT, loc="left", fontsize=12)
    ax_corr.imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax_corr.set_xticks(range(len(symbols)), symbols, rotation=35)
    ax_corr.set_yticks(range(len(symbols)), symbols)
    ax_corr.grid(False)
    ax_corr.set_title("correlation matrix", color=TEXT, loc="left", fontsize=12)
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            ax_corr.text(j, i, f"{corr.iloc[i, j]:.2f}", color=TEXT, ha="center", va="center", fontsize=8)
    save(fig, out)


def plot_07_crypto_ccxt(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(300)
    btc = pd.Series(np.cumprod(1 + rng.normal(0.0007, 0.025, len(idx))), index=idx)
    eth = pd.Series(np.cumprod(1 + rng.normal(0.0008, 0.031, len(idx))), index=idx)
    funding = pd.Series(rng.normal(0.012, 0.035, len(idx)), index=idx).rolling(7).mean()
    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2)
    dashboard_title(fig, "Crypto: Exchange Pricing, Funding and Basis Context", "CCXT-style spot feeds aligned with exchange derivatives diagnostics")
    ax_nav = fig.add_subplot(gs[0, :])
    ax_funding = fig.add_subplot(gs[1, 0])
    ax_state = fig.add_subplot(gs[1, 1])
    for ax in [ax_nav, ax_funding, ax_state]:
        style_ax(ax, "")
    ax_nav.plot(idx, btc / btc.iloc[0] * 100, color=BLUE, lw=2.0, label="Bitcoin spot (CCXT:BTC/USDT)")
    ax_nav.plot(idx, eth / eth.iloc[0] * 100, color=PINK, lw=2.0, label="Ethereum spot (CCXT:ETH/USDT)")
    ax_nav.set_title("normalized spot performance", color=TEXT, loc="left", fontsize=12)
    style_legend(ax_nav)
    ax_funding.bar(idx[-90:], funding[-90:] * 100, color=[GREEN if v > 0 else RED for v in funding[-90:]], alpha=0.72)
    ax_funding.set_title("perpetual funding pressure", color=TEXT, loc="left", fontsize=12)
    ax_funding.set_ylabel("bps")
    state = pd.Series({"spot liquidity": 0.86, "perp funding": 0.54, "open interest": 0.71, "futures basis": 0.63})
    ax_state.barh(state.index, state.values, color=[BLUE, CYAN, PINK, AMBER], alpha=0.9)
    ax_state.set_xlim(0, 1)
    ax_state.set_title("exchange diagnostics state", color=TEXT, loc="left", fontsize=12)
    save(fig, out)


def plot_08_cboe_vix(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(420)
    vix = pd.Series(18 + 6 * np.sin(np.linspace(0, 14, len(idx))) + rng.normal(0, 1.8, len(idx)), index=idx).clip(9, 55)
    vvix = pd.Series(90 + 13 * np.sin(np.linspace(1, 15, len(idx))) + rng.normal(0, 4.5, len(idx)), index=idx)
    spy_dd = -np.maximum(0, (vix - 20) / 180 + rng.normal(0, 0.008, len(idx))).cumsum()
    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
    dashboard_title(fig, "CBOE Volatility: VIX Regime, VVIX and Drawdown Context", "Volatility feeds promoted into a risk regime surface")
    ax_vix = fig.add_subplot(gs[0, 0])
    ax_dd = fig.add_subplot(gs[1, 0], sharex=ax_vix)
    ax_regime = fig.add_subplot(gs[:, 1])
    for ax in [ax_vix, ax_dd, ax_regime]:
        style_ax(ax, "")
    ax_vix.plot(idx, vix, color=CYAN, lw=2.0, label="CBOE Volatility Index (VIX)")
    ax_vix.plot(idx, vvix / 4, color=PINK, lw=1.8, label="CBOE VVIX scaled")
    ax_vix.axhspan(25, 55, color=RED, alpha=0.09)
    ax_vix.set_title("volatility state", color=TEXT, loc="left", fontsize=12)
    style_legend(ax_vix)
    ax_dd.fill_between(idx, spy_dd * 100, 0, color=PINK, alpha=0.22)
    ax_dd.plot(idx, spy_dd * 100, color=PINK, lw=1.4)
    ax_dd.set_title("SPY drawdown overlay conditioned by VIX", color=TEXT, loc="left", fontsize=12)
    ax_dd.set_ylabel("%")
    regimes = pd.DataFrame({"calm": [0.18, 0.12], "normal": [0.55, 0.47], "elevated": [0.27, 0.41]}, index=["VIX", "VVIX"])
    ax_regime.imshow(regimes.values, cmap=LinearSegmentedColormap.from_list("vol", [BLUE, NAVY, RED]), vmin=0, vmax=0.6)
    ax_regime.set_xticks(range(3), regimes.columns)
    ax_regime.set_yticks(range(2), regimes.index)
    ax_regime.grid(False)
    ax_regime.set_title("regime distribution", color=TEXT, loc="left", fontsize=12)
    save(fig, out)


def plot_09_multpl_valuation(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=180, freq="ME")
    cape = pd.Series(24 + np.linspace(0, 9, len(idx)) + 2.5 * np.sin(np.linspace(0, 10, len(idx))) + rng.normal(0, 0.7, len(idx)), index=idx)
    ten_y = pd.Series(2.0 + np.linspace(0, 2.3, len(idx)) + 0.7 * np.sin(np.linspace(0, 8, len(idx))), index=idx)
    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2)
    dashboard_title(fig, "Market Valuation: Shiller P/E and Rates", "Long-horizon valuation context joined to rates for asset-allocation review")
    ax_cape = fig.add_subplot(gs[0, :])
    ax_rate = fig.add_subplot(gs[1, 0])
    ax_pct = fig.add_subplot(gs[1, 1])
    for ax in [ax_cape, ax_rate, ax_pct]:
        style_ax(ax, "")
    ax_cape.plot(idx, cape, color=BLUE, lw=2.0)
    ax_cape.axhspan(30, cape.max() + 2, color=RED, alpha=0.09)
    ax_cape.axhline(cape.quantile(0.75), color=AMBER, lw=1.2, linestyle="--")
    ax_cape.set_title("Shiller CAPE regime", color=TEXT, loc="left", fontsize=12)
    ax_rate.plot(idx, ten_y, color=CYAN, lw=2.0)
    ax_rate.set_title("10Y yield context", color=TEXT, loc="left", fontsize=12)
    metrics = pd.Series({"Shiller CAPE": 0.84, "dividend yield": 0.22, "earnings yield": 0.31, "10Y Treasury": 0.76})
    ax_pct.barh(metrics.index, metrics.values, color=[PINK, GREEN, AMBER, CYAN], alpha=0.9)
    ax_pct.set_xlim(0, 1)
    ax_pct.set_title("valuation component mix", color=TEXT, loc="left", fontsize=12)
    save(fig, out)


def cot_signal_data(stem: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rng = rng_for(stem)
    contracts = ["E-mini S&P 500\n(CFTC:ES)", "WTI Crude Oil\n(CFTC:CL)", "COMEX Gold\n(CFTC:GC)", "10Y Treasury Note\n(CFTC:ZN)"]
    zscores = np.array([1.42, 0.88, 1.11, -1.23]) + rng.normal(0, 0.03, 4)
    percentiles = np.array([82, 68, 74, 9]) + rng.normal(0, 1.2, 4)
    weekly_change = np.array([18420, 9200, 11100, -15600]) + rng.normal(0, 650, 4)
    states = ["Crowded Long", "Long Build", "Long Build", "Crowded Short"]
    return contracts, zscores, percentiles.clip(1, 99), weekly_change, states


def draw_cot_signal_packet(stem: str, out: Path, title: str = "CFTC COT Positioning Signal Packet") -> None:
    contracts, zscores, percentiles, weekly_change, states = cot_signal_data(stem)
    colors = [CYAN if value >= 0 else PINK for value in zscores]
    change_colors = [GREEN if value >= 0 else RED for value in weekly_change]
    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])
    dashboard_title(
        fig,
        title,
        "CFTC COT managed-money positioning - ES / CL / GC / ZN - weekly filing converted into percentile, z-score and tactical state",
    )
    ax_z = fig.add_subplot(gs[0, 0])
    ax_pct = fig.add_subplot(gs[0, 1])
    ax_chg = fig.add_subplot(gs[1, 0])
    ax_state = fig.add_subplot(gs[1, 1])
    for ax in [ax_z, ax_pct, ax_chg, ax_state]:
        style_ax(ax, "")

    x = np.arange(len(contracts))
    ax_z.bar(x, zscores, color=colors, alpha=0.92)
    ax_z.axhline(0, color=TEXT, alpha=0.35, lw=1)
    ax_z.axhline(1, color=RED, alpha=0.55, linestyle="--", lw=1)
    ax_z.axhline(-1, color=GREEN, alpha=0.55, linestyle="--", lw=1)
    ax_z.set_xticks(x, contracts)
    ax_z.set_ylabel("z-score")
    ax_z.set_title("Managed Money Net Positioning Z-Score", color=TEXT, loc="left", fontsize=12)
    for i, value in enumerate(zscores):
        ax_z.text(i, value + (0.10 if value >= 0 else -0.18), f"{value:.1f}", color=TEXT, ha="center", fontsize=9)

    ax_pct.bar(x, percentiles, color=[PINK if p >= 80 or p <= 20 else CYAN for p in percentiles], alpha=0.92)
    ax_pct.axhspan(80, 100, color=RED, alpha=0.08)
    ax_pct.axhspan(0, 20, color=GREEN, alpha=0.07)
    ax_pct.set_ylim(0, 100)
    ax_pct.set_xticks(x, contracts)
    ax_pct.set_ylabel("percentile")
    ax_pct.set_title("Current Positioning Percentile - 3Y Lookback", color=TEXT, loc="left", fontsize=12)
    for i, value in enumerate(percentiles):
        ax_pct.text(i, value + 3, f"{value:.0f}th", color=TEXT, ha="center", fontsize=9)

    ax_chg.bar(x, weekly_change / 1000, color=change_colors, alpha=0.92)
    ax_chg.axhline(0, color=TEXT, alpha=0.35, lw=1)
    ax_chg.set_xticks(x, contracts)
    ax_chg.set_ylabel("net contracts, thousands")
    ax_chg.set_title("1W Change in Managed-Money Net Contracts", color=TEXT, loc="left", fontsize=12)
    for i, value in enumerate(weekly_change / 1000):
        ax_chg.text(i, value + (0.9 if value >= 0 else -1.5), f"{value:+.1f}k", color=TEXT, ha="center", fontsize=9)

    ax_state.axis("off")
    ax_state.set_title("Signal State", color=TEXT, loc="left", fontsize=12, pad=10)
    rows = zip(["ES", "CL", "GC", "ZN"], states, percentiles, weekly_change)
    for i, (code, state, pct, change) in enumerate(rows):
        y = 0.82 - i * 0.19
        state_color = PINK if "Crowded" in state else CYAN
        ax_state.text(0.04, y, code, color=MUTED, fontsize=9, transform=ax_state.transAxes)
        ax_state.text(0.18, y, state, color=state_color, fontsize=13, weight="semibold", transform=ax_state.transAxes)
        ax_state.text(0.18, y - 0.075, f"{pct:.0f}th percentile | {change:+,.0f} net contracts WoW", color=TEXT, fontsize=8.5, transform=ax_state.transAxes)
        ax_state.plot([0.04, 0.94], [y - 0.115, y - 0.115], color=GRID, lw=0.8, alpha=0.7, transform=ax_state.transAxes)
    save(fig, out)


def plot_10_cftc_cot(stem: str, out: Path) -> None:
    draw_cot_signal_packet(stem, out)


def macro_data(stem: str) -> tuple[pd.DatetimeIndex, np.ndarray, pd.Series, pd.Series, np.ndarray]:
    rng = rng_for(stem)
    idx = dates(360)
    curve = np.array([5.12, 4.92, 4.71, 4.38, 4.28, 4.45])
    cpi = pd.Series(3.1 + np.sin(np.linspace(0, 8, len(idx))) * 0.7 + rng.normal(0, 0.08, len(idx)), index=idx)
    unrate = pd.Series(3.9 + np.cos(np.linspace(0, 5, len(idx))) * 0.35 + rng.normal(0, 0.04, len(idx)), index=idx)
    matrix = np.array([[0.72, 0.41, -0.18], [0.55, 0.63, 0.12], [-0.24, 0.31, 0.80]])
    return idx, curve, cpi, unrate, matrix


def plot_03_macro_01(stem: str, out: Path) -> None:
    _, curve, _, _, _ = macro_data(stem)
    tenors = ["3M", "1Y", "2Y", "5Y", "10Y", "30Y"]
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Treasury Curve: Policy Restriction and Duration Risk", "FRED DGS tenor family from 3M to 30Y for rates and scenario context")
    style_ax(ax, "")
    x = np.arange(len(tenors))
    ax.plot(x, curve, color=CYAN, lw=2.8, marker="o", markersize=7)
    ax.fill_between(x, curve, curve.min() - 0.25, color=CYAN, alpha=0.16)
    ax.set_xticks(x, tenors)
    ax.set_ylabel("%")
    save(fig, out)


def plot_03_macro_02(stem: str, out: Path) -> None:
    _, _, cpi, _, _ = macro_data(stem)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Inflation Pressure vs Fed Target", "CPI Urban Consumers YoY (FRED:CPIAUCSL) against the 2 percent policy target")
    style_ax(ax, "")
    ax.plot(cpi.index, cpi, color=PINK, lw=2.2, label="CPI Urban Consumers YoY (FRED:CPIAUCSL)")
    ax.axhline(2.0, color=GREEN, lw=1.5, linestyle="--", label="Federal Reserve target")
    ax.fill_between(cpi.index, 2.0, cpi, where=(cpi > 2.0), color=PINK, alpha=0.12)
    ax.set_ylabel("%")
    style_legend(ax)
    save(fig, out)


def plot_03_macro_03(stem: str, out: Path) -> None:
    _, _, _, unrate, _ = macro_data(stem)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Labor Regime: Unemployment Cycle State", "Civilian Unemployment Rate (FRED:UNRATE) used for cycle-state review")
    style_ax(ax, "")
    ax.plot(unrate.index, unrate, color=AMBER, lw=2.2, label="Civilian Unemployment Rate (FRED:UNRATE)")
    ax.fill_between(unrate.index, unrate.rolling(42).mean(), unrate, color=AMBER, alpha=0.12)
    ax.set_ylabel("%")
    style_legend(ax)
    save(fig, out)


def plot_03_macro_04(stem: str, out: Path) -> None:
    _, _, _, _, matrix = macro_data(stem)
    fig, ax = plt.subplots(figsize=(8.8, 6.3), facecolor=NAVY)
    dashboard_title(fig, "Macro Sensitivity Map: Growth, Inflation and Rates", "Growth composite, CPI pressure and 10Y yield mapped to asset risk")
    style_ax(ax, "")
    ax.grid(False)
    ax.imshow(matrix, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(3), ["Growth composite", "CPI pressure", "10Y yield"])
    ax.set_yticks(range(3), ["Equity beta", "Duration book", "USD exposure"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.2f}", color=TEXT, ha="center", va="center", fontsize=11)
    save(fig, out)


def fundamentals_data() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    peers = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN"]
    pe = np.array([29.4, 34.2, 24.8, 42.7, 27.9, 38.1])
    roe = np.array([1.48, 0.36, 0.29, 0.74, 0.31, 0.22])
    margin = np.array([46.2, 69.1, 57.4, 73.5, 81.3, 48.7])
    return peers, pe, roe, margin


def plot_04_fundamentals_01(stem: str, out: Path) -> None:
    peers, pe, roe, margin = fundamentals_data()
    colors = [BLUE, CYAN, GREEN, PINK, AMBER, RED]
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Peer Valuation: Quality-Adjusted Multiples", "FMP TTM P/E versus ROE with gross-margin scale for IC review")
    style_ax(ax, "")
    ax.scatter(pe, roe, s=margin * 6, color=colors, alpha=0.82, edgecolors=TEXT, linewidths=0.5)
    for peer, x, y in zip(peers, pe, roe):
        ax.text(x + 0.45, y, peer, color=TEXT, fontsize=9)
    ax.set_xlabel("Price / Earnings TTM (FMP)")
    ax.set_ylabel("Return on Equity TTM (FMP)")
    save(fig, out)


def plot_04_fundamentals_02(stem: str, out: Path) -> None:
    peers, pe, _, _ = fundamentals_data()
    colors = [BLUE, CYAN, GREEN, PINK, AMBER, RED]
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Peer Valuation: P/E TTM", "FMP financial-ratio surface normalized across the mega-cap peer set")
    style_ax(ax, "")
    ax.bar(peers, pe, color=colors, alpha=0.92)
    ax.set_ylabel("Price / Earnings TTM")
    save(fig, out)


def plot_04_fundamentals_03(stem: str, out: Path) -> None:
    peers, _, roe, margin = fundamentals_data()
    colors = [BLUE, CYAN, GREEN, PINK, AMBER, RED]
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Peer Quality: Margins and ROE", "FMP gross margin TTM and return on equity for quality review")
    style_ax(ax, "")
    ax.barh(peers, margin, color=colors, alpha=0.9, label="Gross margin TTM (FMP)")
    ax2 = ax.twiny()
    ax2.plot(roe, peers, color=TEXT, marker="o", lw=2.0, label="Return on Equity TTM (FMP)")
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.spines["top"].set_color("#17376f")
    ax2.xaxis.label.set_color(MUTED)
    ax.set_xlabel("gross margin TTM (%)")
    ax2.set_xlabel("ROE")
    save(fig, out)


PORTFOLIO_SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "UNH", "LLY", "AVGO", "V", "MA", "HD", "COST"]


def portfolio_data(stem: str) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    rng = rng_for(stem)
    idx = dates(504)
    factors = rng.normal(size=(len(idx), 4))
    loads = rng.normal(0.25, 0.12, (4, len(PORTFOLIO_SYMBOLS)))
    specific = rng.normal(0.00035, 0.010, (len(idx), len(PORTFOLIO_SYMBOLS)))
    rets = pd.DataFrame(specific + factors @ loads * 0.004, index=idx, columns=PORTFOLIO_SYMBOLS)
    nav = (1 + rets.mean(axis=1)).cumprod()
    risk = (rets.std() / rets.std().sum()).sort_values(ascending=False)
    corr = rets.corr()
    return idx, rets, nav, risk, corr


def plot_06_portfolio_01(stem: str, out: Path) -> None:
    idx, rets, nav, _, _ = portfolio_data(stem)
    rolling_vol = rets.mean(axis=1).rolling(63).std() * np.sqrt(252) * 100
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Portfolio Risk Snapshot: NAV and Realized Volatility", "Fifteen-name equity basket with 63D realized risk state")
    style_ax(ax, "")
    ax.plot(idx, nav, color=BLUE, lw=2.2, label="Model Portfolio NAV")
    ax.fill_between(idx, 1, nav, color=BLUE, alpha=0.12)
    ax2 = ax.twinx()
    ax2.plot(rolling_vol.index, rolling_vol, color=PINK, lw=1.8, label="63D Realized Volatility")
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.spines["right"].set_color("#17376f")
    ax.set_ylabel("NAV")
    ax2.set_ylabel("% annualized", color=MUTED)
    save(fig, out)


def plot_06_portfolio_02(stem: str, out: Path) -> None:
    _, _, _, risk, _ = portfolio_data(stem)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Portfolio Risk Contribution", "Ex-ante volatility contribution across fifteen holdings")
    style_ax(ax, "")
    colors = [PINK if i < 3 else CYAN if i < 8 else BLUE for i in range(len(risk))]
    ax.bar(risk.index, risk.values * 100, color=colors, alpha=0.9)
    ax.set_ylabel("% of risk")
    ax.tick_params(axis="x", rotation=40)
    save(fig, out)


def plot_06_portfolio_03(stem: str, out: Path) -> None:
    _, _, _, _, corr = portfolio_data(stem)
    fig, ax = plt.subplots(figsize=(10.8, 8.2), facecolor=NAVY)
    dashboard_title(fig, "Portfolio Correlation Matrix", "Cross-holding dependency surface for PM risk review")
    style_ax(ax, "")
    ax.grid(False)
    ax.imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(PORTFOLIO_SYMBOLS)), PORTFOLIO_SYMBOLS, rotation=45, ha="right")
    ax.set_yticks(range(len(PORTFOLIO_SYMBOLS)), PORTFOLIO_SYMBOLS)
    for i in range(len(PORTFOLIO_SYMBOLS)):
        for j in range(len(PORTFOLIO_SYMBOLS)):
            if i == j or abs(corr.iloc[i, j]) > 0.35:
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", color=TEXT, ha="center", va="center", fontsize=7)
    save(fig, out)


def crypto_data(stem: str) -> tuple[pd.DatetimeIndex, pd.Series, pd.Series, pd.Series, pd.Series]:
    rng = rng_for(stem)
    idx = dates(300)
    btc = pd.Series(np.cumprod(1 + rng.normal(0.0007, 0.025, len(idx))), index=idx)
    eth = pd.Series(np.cumprod(1 + rng.normal(0.0008, 0.031, len(idx))), index=idx)
    funding = pd.Series(rng.normal(0.012, 0.035, len(idx)), index=idx).rolling(7).mean()
    basis = pd.Series(1.4 + np.sin(np.linspace(0, 10, len(idx))) * 1.1 + rng.normal(0, 0.25, len(idx)), index=idx)
    return idx, btc, eth, funding, basis


def plot_07_crypto_01(stem: str, out: Path) -> None:
    idx, btc, eth, _, _ = crypto_data(stem)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Digital Assets: Spot Performance", "CCXT BTC/USDT and ETH/USDT spot prices normalized for exposure review")
    style_ax(ax, "")
    ax.plot(idx, btc / btc.iloc[0] * 100, color=BLUE, lw=2.2, label="Bitcoin spot (CCXT:BTC/USDT)")
    ax.plot(idx, eth / eth.iloc[0] * 100, color=PINK, lw=2.2, label="Ethereum spot (CCXT:ETH/USDT)")
    ax.set_ylabel("index = 100")
    style_legend(ax)
    save(fig, out)


def plot_07_crypto_02(stem: str, out: Path) -> None:
    _, _, _, funding, _ = crypto_data(stem)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Digital Assets: Perpetual Funding Pressure", "Seven-day funding state for leverage and carry monitoring")
    style_ax(ax, "")
    sample = funding.dropna().tail(120) * 100
    ax.bar(sample.index, sample.values, color=[GREEN if v > 0 else RED for v in sample.values], alpha=0.72)
    ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
    ax.set_ylabel("bps")
    save(fig, out)


def plot_07_crypto_03(stem: str, out: Path) -> None:
    _, _, _, _, basis = crypto_data(stem)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Digital Assets: Futures Basis State", "Annualized futures basis estimate for carry and leverage pressure")
    style_ax(ax, "")
    ax.plot(basis.index, basis, color=CYAN, lw=2.2)
    ax.fill_between(basis.index, 0, basis, where=(basis > 0), color=CYAN, alpha=0.12)
    ax.fill_between(basis.index, 0, basis, where=(basis < 0), color=PINK, alpha=0.12)
    ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
    ax.set_ylabel("% annualized")
    save(fig, out)


def vix_data(stem: str) -> tuple[pd.DatetimeIndex, pd.Series, pd.Series, pd.Series]:
    rng = rng_for(stem)
    idx = dates(420)
    vix = pd.Series(18 + 6 * np.sin(np.linspace(0, 14, len(idx))) + rng.normal(0, 1.8, len(idx)), index=idx).clip(9, 55)
    vvix = pd.Series(90 + 13 * np.sin(np.linspace(1, 15, len(idx))) + rng.normal(0, 4.5, len(idx)), index=idx)
    spy_dd = -np.maximum(0, (vix - 20) / 180 + rng.normal(0, 0.008, len(idx))).cumsum()
    return idx, vix, vvix, spy_dd


def plot_08_vix_01(stem: str, out: Path) -> None:
    idx, vix, vvix, _ = vix_data(stem)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "CBOE Volatility: VIX and VVIX", "CBOE VIX and VVIX pressure promoted into the risk review")
    style_ax(ax, "")
    ax.plot(idx, vix, color=CYAN, lw=2.1, label="CBOE Volatility Index (VIX)")
    ax.plot(idx, vvix / 4, color=PINK, lw=1.9, label="CBOE VVIX scaled")
    ax.axhspan(25, 55, color=RED, alpha=0.09)
    ax.set_ylabel("index")
    style_legend(ax)
    save(fig, out)


def plot_08_vix_02(stem: str, out: Path) -> None:
    idx, vix, _, spy_dd = vix_data(stem)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "CBOE Volatility: SPY Drawdown Overlay", "SPY drawdown conditioned by CBOE VIX pressure")
    style_ax(ax, "")
    ax.fill_between(idx, spy_dd * 100, 0, color=PINK, alpha=0.22)
    ax.plot(idx, spy_dd * 100, color=PINK, lw=1.5, label="SPY drawdown")
    ax2 = ax.twinx()
    ax2.plot(idx, vix, color=CYAN, lw=1.5, alpha=0.75, label="CBOE VIX")
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.spines["right"].set_color("#17376f")
    ax.set_ylabel("drawdown %")
    save(fig, out)


def plot_08_vix_03(stem: str, out: Path) -> None:
    _, vix, vvix, _ = vix_data(stem)
    regimes = pd.DataFrame({"calm": [0.18, 0.12], "normal": [0.55, 0.47], "elevated": [0.27, 0.41]}, index=["VIX", "VVIX"])
    fig, ax = plt.subplots(figsize=(8.8, 6.2), facecolor=NAVY)
    dashboard_title(fig, "CBOE Volatility: Regime Mix", "Share of VIX and VVIX observations across calm, normal and elevated states")
    style_ax(ax, "")
    ax.grid(False)
    ax.imshow(regimes.values, cmap=LinearSegmentedColormap.from_list("vol", [BLUE, NAVY, RED]), vmin=0, vmax=0.6)
    ax.set_xticks(range(3), regimes.columns)
    ax.set_yticks(range(2), regimes.index)
    for i in range(regimes.shape[0]):
        for j in range(regimes.shape[1]):
            ax.text(j, i, f"{regimes.iloc[i, j]:.0%}", color=TEXT, ha="center", va="center", fontsize=11)
    _ = vix, vvix
    save(fig, out)


def multpl_data(stem: str) -> tuple[pd.DatetimeIndex, pd.Series, pd.Series, pd.Series]:
    rng = rng_for(stem)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=180, freq="ME")
    cape = pd.Series(24 + np.linspace(0, 9, len(idx)) + 2.5 * np.sin(np.linspace(0, 10, len(idx))) + rng.normal(0, 0.7, len(idx)), index=idx)
    ten_y = pd.Series(2.0 + np.linspace(0, 2.3, len(idx)) + 0.7 * np.sin(np.linspace(0, 8, len(idx))), index=idx)
    earnings_yield = (1 / cape) * 100
    valuation_spread = earnings_yield - ten_y
    return idx, cape, ten_y, valuation_spread


def plot_09_multpl_01(stem: str, out: Path) -> None:
    idx, cape, _, _ = multpl_data(stem)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Market Valuation: Shiller CAPE", "Multpl Shiller P/E ratio for long-horizon allocation review")
    style_ax(ax, "")
    ax.plot(idx, cape, color=BLUE, lw=2.2)
    ax.axhspan(30, cape.max() + 2, color=RED, alpha=0.09)
    ax.set_ylabel("CAPE")
    save(fig, out)


def plot_09_multpl_02(stem: str, out: Path) -> None:
    idx, _, ten_y, _ = multpl_data(stem)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Market Valuation: 10Y Treasury Context", "FRED:DGS10 10-Year Treasury Yield beside valuation signals")
    style_ax(ax, "")
    ax.plot(idx, ten_y, color=CYAN, lw=2.2)
    ax.fill_between(idx, ten_y.rolling(24).mean(), ten_y, color=CYAN, alpha=0.12)
    ax.set_ylabel("%")
    save(fig, out)


def plot_09_multpl_03(stem: str, out: Path) -> None:
    idx, _, _, spread = multpl_data(stem)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Market Valuation: Earnings-Yield Spread", "S&P earnings yield minus FRED:DGS10 for asset-allocation context")
    style_ax(ax, "")
    ax.plot(idx, spread, color=PINK, lw=2.2)
    ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
    ax.fill_between(idx, 0, spread, where=(spread >= 0), color=GREEN, alpha=0.10)
    ax.fill_between(idx, 0, spread, where=(spread < 0), color=PINK, alpha=0.12)
    ax.set_ylabel("spread, pp")
    save(fig, out)


def plot_11_sec_01(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(300)
    price = pd.Series(np.cumprod(1 + rng.normal(0.00045, 0.014, len(idx))) * 175, index=idx)
    filing_dates = idx[[48, 112, 176, 241]]
    filing_labels = ["10-K", "10-Q", "8-K", "10-Q"]
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "AAPL Price Reaction Around SEC Events", "SEC 10-K, 10-Q and 8-K filing dates over Tiingo adjusted close")
    style_ax(ax, "")
    ax.plot(idx, price, color=CYAN, lw=2.2, label="AAPL adjusted close (Tiingo/EOD)")
    ax.fill_between(idx, price.rolling(21).mean(), price, color=CYAN, alpha=0.12)
    for event_date, label in zip(filing_dates, filing_labels):
        ax.axvline(event_date, color=PINK, lw=1.3, alpha=0.85)
        ax.text(event_date, price.max() * 0.985, label, color=TEXT, fontsize=8, rotation=90, va="top", ha="right")
    ax.set_ylabel("price")
    style_legend(ax)
    save(fig, out)


def plot_12_finra_01(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(240)
    ratio = pd.Series(0.23 + np.sin(np.linspace(0, 11, len(idx))) * 0.06 + rng.normal(0, 0.018, len(idx)), index=idx).clip(0.05, 0.55)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Crowding Pressure: FINRA Short-Volume Ratio", "FINRA daily short volume ratio for crowded-trade review")
    style_ax(ax, "")
    ax.plot(idx, ratio * 100, color=PINK, lw=2.0)
    ax.fill_between(idx, ratio * 100, ratio.rolling(63).mean() * 100, color=PINK, alpha=0.12)
    ax.set_ylabel("%")
    save(fig, out)


def plot_12_finra_02(stem: str, out: Path) -> None:
    symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
    days = pd.Series([1.2, 0.9, 1.8, 1.1, 1.5, 0.8, 2.6], index=symbols)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Crowding Pressure: Days-to-Cover", "Short-interest context and estimated cover time by symbol")
    style_ax(ax, "")
    ax.bar(symbols, days, color=[CYAN, BLUE, PINK, GREEN, AMBER, CYAN, RED], alpha=0.9)
    ax.set_ylabel("days")
    save(fig, out)


def plot_12_finra_03(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    z = rng.normal(0, 0.55, (6, 6))
    fig, ax = plt.subplots(figsize=(9.5, 6.2), facecolor=NAVY)
    dashboard_title(fig, "Crowding Map: Sector x Liquidity Bucket", "Short pressure by sector and liquidity bucket for capacity review")
    style_ax(ax, "")
    ax.grid(False)
    ax.imshow(z, cmap=CORR_CMAP, vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(6), ["mega", "large", "mid", "small", "high beta", "low vol"], rotation=35, ha="right")
    ax.set_yticks(range(6), ["tech", "comm", "cons", "fin", "energy", "health"])
    save(fig, out)


def plot_13_openfigi_01(stem: str, out: Path) -> None:
    fields = [
        ("ticker", "AAPL"),
        ("name", "Apple Inc."),
        ("composite FIGI", "BBG000B9XRY4"),
        ("share class FIGI", "BBG001S5N8V8"),
        ("security type", "Common Stock"),
        ("exchange", "XNAS"),
        ("currency", "USD"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Security Master: AAPL Identifier Resolution", "OpenFIGI composite and share-class identifiers for cross-vendor reconciliation")
    style_ax(ax, "")
    ax.axis("off")
    y = 0.82
    for label, value in fields:
        ax.text(0.08, y, label.upper(), color=MUTED, fontsize=9, transform=ax.transAxes)
        ax.text(0.38, y, value, color=TEXT, fontsize=15, weight="semibold", transform=ax.transAxes)
        ax.plot([0.08, 0.92], [y - 0.045, y - 0.045], color=GRID, lw=0.8, alpha=0.65, transform=ax.transAxes)
        y -= 0.105
    save(fig, out)


def plot_14_actions_01(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    x = dates(420)
    close = pd.Series(np.cumprod(1 + rng.normal(0.00045, 0.012, len(x))) * 180, index=x)
    ratio = pd.Series(0.96, index=x)
    ratio.iloc[180:] = 0.985
    ratio.iloc[310:] = 1.0
    adjusted = close * ratio
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Corporate Actions: Raw vs Adjusted Price", "Split and dividend adjustment semantics for PIT replay and total-return research")
    style_ax(ax, "")
    ax.plot(close.index, close, color=CYAN, linewidth=2.1, label="Raw close")
    ax.plot(adjusted.index, adjusted, color=PINK, linewidth=2.1, label="Split/dividend-adjusted close")
    style_legend(ax)
    ax.set_ylabel("price")
    save(fig, out)


def plot_15_domain_01(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(252)
    price = pd.Series(np.cumprod(1 + rng.normal(0.00052, 0.0135, len(idx))) * 190, index=idx)
    pe = pd.Series(27.5 + np.sin(np.linspace(0, 8, len(idx))) * 3.2 + rng.normal(0, 0.45, len(idx)), index=idx)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "AAPL Equity Packet: Pricing + Fundamentals", "One ticker routed into adjusted price and P/E TTM data for research review")
    style_ax(ax, "")
    ax.plot(idx, price, color=CYAN, lw=2.1, label="AAPL adjusted close")
    ax.fill_between(idx, price.rolling(21).mean(), price, color=CYAN, alpha=0.10)
    ax.set_ylabel("price")
    ax2 = ax.twinx()
    ax2.plot(idx, pe, color=PINK, lw=1.8, label="AAPL P/E TTM")
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.spines["right"].set_color("#17376f")
    ax2.set_ylabel("P/E TTM", color=MUTED)
    save(fig, out)


def plot_16_sources_01(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=96, freq="ME")
    cpi = pd.Series(2.4 + np.sin(np.linspace(0, 10, len(idx))) * 0.9 + rng.normal(0, 0.10, len(idx)), index=idx)
    fed = pd.Series(1.8 + np.linspace(0, 3.0, len(idx)) + np.sin(np.linspace(0, 6, len(idx))) * 0.55, index=idx).clip(0, 5.6)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Inflation vs Policy Rate", "FRED:CPIAUCSL CPI YoY aligned with FRED:FEDFUNDS policy-rate path")
    style_ax(ax, "")
    ax.plot(idx, cpi, color=PINK, lw=2.1, label="CPI Urban Consumers YoY (FRED:CPIAUCSL)")
    ax.plot(idx, fed, color=CYAN, lw=2.1, label="Effective Fed Funds Rate (FRED:FEDFUNDS)")
    ax.axhline(2.0, color=GREEN, lw=1.2, linestyle="--", alpha=0.8)
    ax.set_ylabel("%")
    style_legend(ax)
    save(fig, out)


def plot_16_sources_02(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=72, freq="ME")
    us_pmi = pd.Series(51 + np.sin(np.linspace(0, 8, len(idx))) * 4.2 + rng.normal(0, 0.65, len(idx)), index=idx)
    eu_pmi = pd.Series(49 + np.sin(np.linspace(1, 9, len(idx))) * 3.4 + rng.normal(0, 0.70, len(idx)), index=idx)
    china_pmi = pd.Series(50 + np.sin(np.linspace(2, 10, len(idx))) * 2.8 + rng.normal(0, 0.55, len(idx)), index=idx)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Global Growth Pulse", "Regional PMI-style growth series for cross-market allocation context")
    style_ax(ax, "")
    ax.plot(idx, us_pmi, color=BLUE, lw=2.0, label="US manufacturing PMI composite")
    ax.plot(idx, eu_pmi, color=CYAN, lw=2.0, label="Euro area PMI composite")
    ax.plot(idx, china_pmi, color=PINK, lw=2.0, label="China PMI composite")
    ax.axhline(50, color=TEXT, lw=1.0, alpha=0.35)
    ax.set_ylabel("PMI")
    style_legend(ax)
    save(fig, out)


def plot_16_sources_03(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(360)
    oil = pd.Series(np.cumprod(1 + rng.normal(0.00020, 0.018, len(idx))) * 82, index=idx)
    breakeven = pd.Series(2.25 + np.sin(np.linspace(0, 9, len(idx))) * 0.32 + rng.normal(0, 0.035, len(idx)), index=idx)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Energy and Inflation Expectations", "WTI crude oil beside 10Y breakeven-style inflation expectations")
    style_ax(ax, "")
    ax.plot(idx, oil, color=AMBER, lw=2.0, label="WTI crude oil (FRED:DCOILWTICO)")
    ax.set_ylabel("oil")
    ax2 = ax.twinx()
    ax2.plot(idx, breakeven, color=CYAN, lw=1.8, label="10Y breakeven inflation (FRED:T10YIE)")
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.spines["right"].set_color("#17376f")
    ax2.set_ylabel("%", color=MUTED)
    save(fig, out)


def plot_17_vol_01(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    x = dates(360)
    vix = pd.Series(18 + 5 * np.sin(np.linspace(0, 13, len(x))) + rng.normal(0, 1.5, len(x)), index=x).clip(9, 55)
    skew = pd.Series(125 + 9 * np.cos(np.linspace(0, 10, len(x))) + rng.normal(0, 3, len(x)), index=x)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Derivatives: VIX and SKEW", "CBOE volatility and tail-risk context in one route family")
    style_ax(ax, "")
    ax.plot(vix.index, vix, color=CYAN, linewidth=2.1, label="CBOE Volatility Index (VIX)")
    ax.plot(skew.index, (skew - 100) / 2, color=PINK, linewidth=2.1, label="CBOE SKEW scaled")
    ax.set_ylabel("index")
    style_legend(ax)
    save(fig, out)


def plot_17_vol_02(stem: str, out: Path) -> None:
    expiries = ["1W", "1M", "3M", "6M", "1Y"]
    term = np.array([18.2, 19.4, 21.1, 22.0, 23.3])
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Derivatives: Volatility Term Structure", "Expiry curve for option-overlay and hedge-tenor decisions")
    style_ax(ax, "")
    ax.plot(expiries, term, color=CYAN, lw=2.8, marker="o")
    ax.fill_between(range(len(expiries)), term, term.min() - 1.0, color=CYAN, alpha=0.12)
    ax.set_ylabel("implied vol")
    save(fig, out)


def plot_17_vol_03(stem: str, out: Path) -> None:
    strikes = np.linspace(80, 120, 9)
    expiries = ["1M", "3M", "6M", "1Y"]
    surface = np.array([[31, 27, 24, 23, 22, 22, 23, 25, 28], [29, 25, 23, 22, 21, 21, 22, 24, 27], [27, 24, 22, 21, 20, 20, 21, 23, 25], [26, 23, 21, 20, 19, 19, 20, 22, 24]])
    fig, ax = plt.subplots(figsize=(9.8, 6.4), facecolor=NAVY)
    dashboard_title(fig, "Derivatives: Volatility Surface", "Strike and expiry context for Greeks, collars and overlay decisions")
    style_ax(ax, "")
    ax.grid(False)
    ax.imshow(surface, cmap=CORR_CMAP, aspect="auto")
    ax.set_xticks(range(len(strikes)), [f"{s:.0f}" for s in strikes])
    ax.set_yticks(range(len(expiries)), expiries)
    ax.set_xlabel("moneyness")
    save(fig, out)


def plot_18_index_01(stem: str, out: Path) -> None:
    labels = ["liquidity", "float", "size", "profitability", "data quality", "tradability"]
    values = [0.92, 0.86, 0.88, 0.71, 0.96, 0.90]
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Index Constituents: Investability Scores", "Constituent metadata converted into liquidity, float and tradability checks")
    style_ax(ax, "")
    ax.bar(labels, values, color=[BLUE, CYAN, PINK, GREEN, AMBER, RED], alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    save(fig, out)


def plot_18_index_02(stem: str, out: Path) -> None:
    sectors = pd.Series({"Tech": 31, "Financials": 13, "Health": 12, "Comm": 9, "Industrials": 8, "Consumer": 11, "Other": 16})
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Index Constituents: Sector Weights", "Benchmark sector context for universe construction")
    style_ax(ax, "")
    ax.barh(sectors.index, sectors.values, color=[BLUE, CYAN, PINK, GREEN, AMBER, RED, BLUE], alpha=0.9)
    ax.set_xlabel("% benchmark")
    save(fig, out)


def plot_18_index_03(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    x = rng.lognormal(3.1, 0.55, 80)
    y = 0.35 + np.log1p(x) / 8 + rng.normal(0, 0.05, 80)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "Index Constituents: Liquidity Filter", "Dollar ADV and investability score for candidate names")
    style_ax(ax, "")
    ax.scatter(x, y, color=CYAN, alpha=0.78, edgecolors=TEXT, linewidths=0.35)
    ax.set_xscale("log")
    ax.set_xlabel("63D dollar ADV estimate")
    ax.set_ylabel("investability score")
    save(fig, out)


def plot_19_lineage_01(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(260)
    price = pd.Series(np.cumprod(1 + rng.normal(0.00050, 0.014, len(idx))) * 185, index=idx)
    pe = pd.Series(28 + np.sin(np.linspace(0, 7, len(idx))) * 2.7 + rng.normal(0, 0.35, len(idx)), index=idx)
    event_dates = idx[[55, 132, 205]]
    event_labels = ["10-Q", "Form 4", "8-K"]
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NAVY)
    dashboard_title(fig, "AAPL Evidence Timeline: Price, Valuation and Filings", "Adjusted price, P/E TTM and SEC event labels kept together for audit-ready research output")
    style_ax(ax, "")
    ax.plot(idx, price, color=CYAN, lw=2.1, label="AAPL adjusted close")
    ax.set_ylabel("price")
    for event_date, label in zip(event_dates, event_labels):
        ax.axvline(event_date, color=PINK, lw=1.2, alpha=0.85)
        ax.text(event_date, price.max() * 0.99, label, color=TEXT, fontsize=8, rotation=90, va="top", ha="right")
    ax2 = ax.twinx()
    ax2.plot(idx, pe, color=GREEN, lw=1.7, alpha=0.9, label="AAPL P/E TTM")
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.spines["right"].set_color("#17376f")
    ax2.set_ylabel("P/E TTM", color=MUTED)
    save(fig, out)


def plot_nav(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    s = random_walk(rng)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(s.index, s / s.iloc[0], color=BLUE, linewidth=2.4)
    ax.fill_between(s.index, 1, s / s.iloc[0], color=BLUE, alpha=0.16)
    ax.set_ylabel("normalized value")
    save(fig, out)


def plot_multi_nav(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    for label, color, drift, vol in [
        ("AAPL adjusted close", BLUE, 0.00055, 0.014),
        ("MSFT adjusted close", CYAN, 0.00050, 0.012),
        ("NVDA adjusted close", PINK, 0.00078, 0.021),
    ]:
        s = random_walk(rng, drift=drift, vol=vol)
        ax.plot(s.index, s / s.iloc[0], color=color, linewidth=2.1, label=label)
    ax.legend(facecolor=PANEL, edgecolor="#17376f", labelcolor=TEXT)
    ax.set_ylabel("normalized value")
    save(fig, out)


def plot_bar(stem: str, title: str, out: Path, labels: list[str] | None = None) -> None:
    rng = rng_for(stem)
    labels = labels or ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    values = np.abs(rng.normal(1.0, 0.35, len(labels)))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.bar(labels, values, color=[BLUE, CYAN, PINK, GREEN, AMBER, RED][: len(labels)], alpha=0.92)
    ax.set_ylabel("PM review score")
    save(fig, out)


def plot_heatmap(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    labels = ["SPY equity beta", "TLT duration", "GLD gold", "UUP USD", "BTC digital asset"]
    x = rng.normal(size=(420, len(labels)))
    corr = np.corrcoef(x.T)
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    style_ax(ax, title)
    ax.grid(False)
    ax.imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color=TEXT, fontsize=9)
    save(fig, out)


def plot_drawdown(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    nav = random_walk(rng, vol=0.014)
    dd = nav / nav.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(dd.index, dd * 100, color=PINK, linewidth=2.2)
    ax.fill_between(dd.index, dd * 100, 0, color=PINK, alpha=0.18)
    ax.set_ylabel("underwater drawdown (%)")
    save(fig, out)


def plot_yield_curve(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    tenors = ["3M", "1Y", "2Y", "5Y", "10Y", "30Y"]
    base = np.array([4.95, 4.75, 4.55, 4.38, 4.30, 4.42])
    curve = base + rng.normal(0, 0.05, len(base))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(tenors, curve, color=CYAN, linewidth=2.8, marker="o", markersize=7)
    ax.set_ylabel("Treasury yield (%)")
    save(fig, out)


def plot_payoff(stem: str, title: str, out: Path) -> None:
    spot = np.linspace(70, 130, 121)
    payoff = np.minimum(np.maximum(spot - 100, -12), 16) + 2.5
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(spot, payoff, color=GREEN, linewidth=2.5)
    ax.axhline(0, color=TEXT, alpha=0.35, linewidth=1.0)
    ax.set_xlabel("underlying at expiry (% of spot)")
    ax.set_ylabel("option overlay payoff")
    save(fig, out)


def plot_scatter(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    x = rng.lognormal(18, 0.7, 38) / 1e6
    y = rng.normal(0.18, 0.07, 38)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.scatter(x, y, s=58, color=CYAN, alpha=0.82, edgecolors=TEXT, linewidths=0.35)
    ax.set_xscale("log")
    ax.set_xlabel("63D dollar ADV / capacity estimate")
    ax.set_ylabel("risk, crowding or implementation score")
    save(fig, out)


def plot_event_curve(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    x = np.arange(-10, 43)
    y = np.tanh(x / 18) * 0.05 + rng.normal(0, 0.004, len(x))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(x, y * 100, color=AMBER, linewidth=2.4)
    ax.axvline(0, color=TEXT, alpha=0.35, linewidth=1.0)
    ax.set_xlabel("trading days from earnings, SEC or regulatory event")
    ax.set_ylabel("average forward return (%)")
    save(fig, out)


def plot_monte_carlo(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    paths = np.cumprod(1 + rng.normal(0.00045, 0.012, (252, 600)), axis=0)
    bands = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    x = np.arange(paths.shape[0])
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.fill_between(x, bands[0], bands[4], color=BLUE, alpha=0.14)
    ax.fill_between(x, bands[1], bands[3], color=CYAN, alpha=0.22)
    ax.plot(x, bands[2], color=TEXT, linewidth=2.1)
    ax.set_xlabel("trading days")
    ax.set_ylabel("simulated portfolio NAV")
    save(fig, out)


def plot_grid(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    z = rng.normal(0.9, 0.35, (7, 7))
    z += np.linspace(-0.2, 0.3, 7)[None, :]
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    style_ax(ax, title)
    ax.grid(False)
    ax.imshow(z, cmap="magma")
    ax.set_xlabel("slow signal window")
    ax.set_ylabel("fast signal window")
    save(fig, out)


def plot_evidence_packet(stem: str, title: str, out: Path) -> None:
    labels = ["Adjusted OHLCV", "TTM ratios", "SEC filings", "Form 4", "OpenFIGI", "FINRA shorts", "CBOE vol"]
    values = [0.92, 0.78, 0.64, 0.48, 0.86, 0.52, 0.71]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title, )
    ax.bar(labels, values, color=[BLUE, CYAN, PINK, GREEN, AMBER, RED, "#8ab4ff"], alpha=0.92)
    ax.set_ylabel("evidence strength score")
    ax.set_ylim(0, 1.05)
    save(fig, out)


def plot_scenario_bars(stem: str, title: str, out: Path) -> None:
    labels = ["+100bp inflation shock", "hard landing", "USD squeeze", "risk-on rebound"]
    values = [-3.8, -5.6, -2.9, 4.4]
    colors = [RED if value < 0 else GREEN for value in values]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.bar(labels, values, color=colors, alpha=0.9)
    ax.axhline(0, color=TEXT, alpha=0.35, linewidth=1)
    ax.set_ylabel("estimated portfolio impact (%)")
    save(fig, out)


def plot_regime_timeline(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    x = dates(420)
    growth = np.sin(np.linspace(0, 10, len(x))) + rng.normal(0, 0.12, len(x))
    inflation = np.cos(np.linspace(0, 8, len(x))) + rng.normal(0, 0.12, len(x))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(x, growth, color=CYAN, linewidth=2.1, label="Growth composite (PMI / activity)")
    ax.plot(x, inflation, color=PINK, linewidth=2.1, label="Inflation pressure (CPI / breakeven)")
    ax.fill_between(x, -2, 2, where=(growth < 0), color=RED, alpha=0.08)
    ax.fill_between(x, -2, 2, where=(inflation > 0), color=AMBER, alpha=0.08)
    ax.text(x[45], 1.65, "Rising Rate Regime\npost-tightening", color=AMBER, fontsize=8, va="top")
    ax.text(x[245], -1.55, "Growth Slowdown\nrisk-budget review", color=TEXT, fontsize=8, va="bottom")
    ax.legend(facecolor=PANEL, edgecolor="#17376f", labelcolor=TEXT)
    ax.set_ylabel("regime z-score")
    save(fig, out)


def plot_inflation_monitor(stem: str, title: str, out: Path) -> None:
    labels = ["XLE energy", "XLK technology", "XLF financials", "XLU utilities", "XLP staples", "XLY discretionary"]
    values = [0.82, -0.21, 0.18, -0.33, -0.08, 0.27]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.bar(labels, values, color=[GREEN if value > 0 else PINK for value in values], alpha=0.92)
    ax.axhline(0, color=TEXT, alpha=0.35, linewidth=1)
    ax.set_ylabel("inflation beta estimate")
    save(fig, out)


def plot_positioning_dashboard(stem: str, title: str, out: Path) -> None:
    draw_cot_signal_packet(stem, out, human_title(title))


def plot_weight_bar(stem: str, title: str, out: Path) -> None:
    labels = ["SPY equity", "TLT duration", "GLD gold", "DBC commodities", "UUP USD"]
    values = [0.24, 0.28, 0.20, 0.16, 0.12]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.bar(labels, values, color=[BLUE, CYAN, PINK, GREEN, AMBER], alpha=0.92)
    ax.set_ylabel("target portfolio weight")
    ax.set_ylim(0, 0.34)
    save(fig, out)


def plot_risk_contributors(stem: str, title: str, out: Path) -> None:
    labels = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "JPM"]
    values = [0.31, 0.18, 0.16, 0.12, 0.10, 0.08, 0.05]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.barh(labels[::-1], values[::-1], color=[BLUE, CYAN, PINK, GREEN, AMBER, RED, "#8ab4ff"][::-1], alpha=0.92)
    ax.set_xlabel("ex-ante risk contribution")
    save(fig, out)


def plot_flow_mosaic(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(260)
    institutional_flow = pd.Series(rng.normal(0.02, 0.18, len(idx)), index=idx).rolling(10).mean().fillna(0).cumsum()
    congress_flow = pd.Series(rng.normal(0.01, 0.10, len(idx)), index=idx).rolling(15).mean().fillna(0).cumsum()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(idx, institutional_flow, color=CYAN, linewidth=2.1, label="Institutional 13F net flow signal (SEC:13F-HR)")
    ax.plot(idx, congress_flow, color=PINK, linewidth=2.1, label="Congress trade flow signal (House/Senate)")
    ax.axhline(0, color=TEXT, alpha=0.35, linewidth=1)
    ax.set_ylabel("cumulative flow signal")
    style_legend(ax)
    save(fig, out)


def plot_investment_evidence_timeline(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(300)
    price = pd.Series(np.cumprod(1 + rng.normal(0.00055, 0.014, len(idx))) * 180, index=idx)
    events = idx[[45, 96, 154, 219, 262]]
    labels = ["SEC 10-K", "Form 4 insider", "FMP ratio update", "CBOE VIX spike", "SEC 13F-HR"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(idx, price, color=CYAN, linewidth=2.2, label="AAPL adjusted close")
    ax.fill_between(idx, price.rolling(21).mean(), price, color=CYAN, alpha=0.10)
    for event_date, label in zip(events, labels):
        ax.axvline(event_date, color=PINK, lw=1.2, alpha=0.8)
        ax.text(event_date, price.max() * 0.985, label, color=TEXT, fontsize=8, rotation=90, va="top", ha="right")
    ax.set_ylabel("price")
    save(fig, out)


def plot_adjustment_semantics(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    x = dates(420)
    close = pd.Series(np.cumprod(1 + rng.normal(0.00045, 0.012, len(x))) * 180, index=x)
    ratio = pd.Series(0.96, index=x)
    ratio.iloc[180:] = 0.985
    ratio.iloc[310:] = 1.0
    adjusted = close * ratio
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(close.index, close, color=CYAN, linewidth=2.1, label="Raw close")
    ax.plot(adjusted.index, adjusted, color=PINK, linewidth=2.1, label="Split/dividend-adjusted close")
    ax.legend(facecolor=PANEL, edgecolor="#17376f", labelcolor=TEXT)
    ax.set_ylabel("price")
    save(fig, out)


def plot_domain_contract(stem: str, title: str, out: Path) -> None:
    labels = ["domains", "aliases", "described", "scoped", "callable"]
    values = [540, 120, 5, 5, 4]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.bar(labels, values, color=[BLUE, CYAN, PINK, GREEN, AMBER], alpha=0.92)
    ax.set_ylabel("route contract objects")
    save(fig, out)


def plot_macro_sources(stem: str, title: str, out: Path) -> None:
    labels = ["FRED", "IMF", "OECD", "WorldBank", "DBnomics", "Eurostat"]
    values = [1.0, 0.82, 0.74, 0.86, 0.68, 0.59]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.bar(labels, values, color=[BLUE, CYAN, PINK, GREEN, AMBER, RED], alpha=0.92)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("macro dataset usability score")
    save(fig, out)


def plot_vol_surface_context(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    x = dates(360)
    vix = pd.Series(18 + 5 * np.sin(np.linspace(0, 13, len(x))) + rng.normal(0, 1.5, len(x)), index=x).clip(9, 55)
    skew = pd.Series(125 + 9 * np.cos(np.linspace(0, 10, len(x))) + rng.normal(0, 3, len(x)), index=x)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(vix.index, vix, color=CYAN, linewidth=2.1, label="CBOE VIX")
    ax.plot(skew.index, (skew - 100) / 2, color=PINK, linewidth=2.1, label="CBOE SKEW scaled")
    ax.legend(facecolor=PANEL, edgecolor="#17376f", labelcolor=TEXT)
    ax.set_ylabel("volatility context index")
    save(fig, out)


def plot_universe_build(stem: str, title: str, out: Path) -> None:
    labels = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM"]
    values = [0.72, 0.69, 0.84, 0.58, 0.61, 0.64, 0.47, 0.43]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.bar(labels, values, color=[BLUE, CYAN, PINK, GREEN, AMBER, RED, "#8ab4ff", "#b6e880"], alpha=0.92)
    ax.set_ylim(0, 1)
    ax.set_ylabel("investability score")
    save(fig, out)


def plot_lineage_packet(stem: str, title: str, out: Path) -> None:
    labels = ["prices", "ratios", "filings", "identity"]
    values = [252, 1, 10, 1]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.bar(labels, values, color=[BLUE, CYAN, PINK, GREEN], alpha=0.92)
    ax.set_yscale("log")
    ax.set_ylabel("rows / evidence objects")
    save(fig, out)


def plot_usd_liquidity_mosaic(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(520)
    walcl = pd.Series(8.7 + np.cumsum(rng.normal(-0.0007, 0.011, len(idx))), index=idx)
    tga = pd.Series(0.55 + np.sin(np.linspace(0, 11, len(idx))) * 0.18 + rng.normal(0, 0.025, len(idx)), index=idx)
    rrp = pd.Series(1.6 - np.linspace(0, 0.95, len(idx)) + np.sin(np.linspace(0, 7, len(idx))) * 0.22 + rng.normal(0, 0.025, len(idx)), index=idx).clip(0.05, None)
    liquidity = ((walcl.pct_change(26) - tga.pct_change(26) - rrp.pct_change(26)).rolling(8).mean() * 100).dropna()
    vix = pd.Series(18 + np.sin(np.linspace(0, 14, len(idx))) * 5.5 + rng.normal(0, 1.4, len(idx)), index=idx).clip(10, 45)
    assets = pd.DataFrame({
        "SPY equity": np.cumprod(1 + rng.normal(0.00045, 0.011, len(idx))),
        "TLT duration": np.cumprod(1 + rng.normal(0.00008, 0.010, len(idx))),
        "GLD gold": np.cumprod(1 + rng.normal(0.00022, 0.009, len(idx))),
        "UUP dollar": np.cumprod(1 + rng.normal(0.00006, 0.005, len(idx))),
        "BTC crypto": np.cumprod(1 + rng.normal(0.00072, 0.026, len(idx))),
    }, index=idx)
    sens = pd.Series({"SPY equity": 0.42, "TLT duration": -0.18, "GLD gold": 0.16, "UUP dollar": -0.31, "BTC crypto": 0.58})

    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2)
    dashboard_title(fig, human_title(title), "Fed liquidity, rates, volatility and cross-asset prices normalized into one data packet")
    ax_liq = fig.add_subplot(gs[0, 0])
    ax_asset = fig.add_subplot(gs[0, 1])
    ax_sens = fig.add_subplot(gs[1, 0])
    ax_comp = fig.add_subplot(gs[1, 1])
    for ax in [ax_liq, ax_asset, ax_sens, ax_comp]:
        style_ax(ax, "")

    ax_liq.plot(liquidity.index, liquidity, color=CYAN, lw=2.1, label="USD liquidity impulse")
    ax_liq.axhline(0, color=TEXT, alpha=0.35, lw=1)
    ax_liq.fill_between(liquidity.index, 0, liquidity, where=(liquidity >= 0), color=GREEN, alpha=0.10)
    ax_liq.fill_between(liquidity.index, 0, liquidity, where=(liquidity < 0), color=PINK, alpha=0.12)
    ax_liq2 = ax_liq.twinx()
    ax_liq2.plot(vix.index, vix, color=PINK, lw=1.4, alpha=0.75, label="CBOE VIX")
    ax_liq2.tick_params(colors=MUTED, labelsize=9)
    ax_liq2.spines["right"].set_color("#17376f")
    ax_liq.set_title("liquidity impulse and volatility", color=TEXT, loc="left", fontsize=12)

    for col, color in zip(assets.columns, [BLUE, CYAN, GREEN, AMBER, PINK]):
        ax_asset.plot(assets.index, assets[col] / assets[col].iloc[0] * 100, color=color, lw=1.7, label=col)
    ax_asset.set_title("cross-asset normalized price paths", color=TEXT, loc="left", fontsize=12)
    ax_asset.set_ylabel("index = 100")
    style_legend(ax_asset)

    ax_sens.bar(sens.index, sens.values, color=[GREEN if value > 0 else PINK for value in sens], alpha=0.92)
    ax_sens.axhline(0, color=TEXT, alpha=0.35, lw=1)
    ax_sens.tick_params(axis="x", rotation=25)
    ax_sens.set_title("252D beta to liquidity impulse", color=TEXT, loc="left", fontsize=12)

    comp = pd.DataFrame({"Fed balance sheet (WALCL)": walcl, "Treasury account (WTREGEN)": tga, "Reverse repo (RRPONTSYD)": rrp})
    for col, color in zip(comp.columns, [BLUE, AMBER, PINK]):
        normalized = (comp[col] / comp[col].iloc[0] - 1) * 100
        ax_comp.plot(comp.index, normalized, color=color, lw=1.8, label=col)
    ax_comp.set_title("liquidity components", color=TEXT, loc="left", fontsize=12)
    ax_comp.set_ylabel("% since start")
    style_legend(ax_comp)
    save(fig, out)


def plot_earnings_options_reaction(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    event_days = np.arange(-10, 22)
    event_curve = pd.Series(np.tanh(event_days / 16) * 0.045 + rng.normal(0, 0.004, len(event_days)), index=event_days)
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
    implied = pd.Series([4.6, 4.1, 7.8, 5.2, 4.9, 5.5], index=symbols)
    realized = implied * pd.Series([0.82, 1.10, 1.34, 0.76, 0.91, 1.18], index=symbols)
    surprise = pd.Series([3.2, 1.6, 8.9, -0.8, 2.4, 4.1], index=symbols)
    fwd = pd.Series([2.1, 1.4, 6.8, -1.1, 0.9, 3.2], index=symbols)
    idx = dates(260)
    vix = pd.Series(17 + np.sin(np.linspace(0, 11, len(idx))) * 5 + rng.normal(0, 1.2, len(idx)), index=idx).clip(10, 42)

    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2)
    dashboard_title(fig, human_title(title), "Earnings surprise, option-implied move, VIX and event-window price reaction")
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_move = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_vix = fig.add_subplot(gs[1, 1])
    for ax in [ax_curve, ax_move, ax_scatter, ax_vix]:
        style_ax(ax, "")

    ax_curve.plot(event_curve.index, event_curve * 100, color=CYAN, lw=2.3)
    ax_curve.axvline(0, color=TEXT, alpha=0.35, lw=1)
    ax_curve.fill_between(event_curve.index, 0, event_curve * 100, color=CYAN, alpha=0.12)
    ax_curve.set_title("average event-window return", color=TEXT, loc="left", fontsize=12)
    ax_curve.set_xlabel("trading days from earnings")
    ax_curve.set_ylabel("%")

    x = np.arange(len(symbols))
    ax_move.bar(x - 0.18, implied.values, width=0.36, color=BLUE, alpha=0.88, label="option implied move")
    ax_move.bar(x + 0.18, realized.values, width=0.36, color=PINK, alpha=0.88, label="realized 1D move")
    ax_move.set_xticks(x, symbols)
    ax_move.set_ylabel("%")
    ax_move.set_title("implied vs realized move", color=TEXT, loc="left", fontsize=12)
    style_legend(ax_move)

    ax_scatter.scatter(surprise, fwd, s=90, color=[BLUE, CYAN, PINK, GREEN, AMBER, RED], edgecolors=TEXT, linewidths=0.4)
    for symbol in symbols:
        ax_scatter.text(surprise[symbol] + 0.12, fwd[symbol], symbol, color=TEXT, fontsize=9)
    ax_scatter.axhline(0, color=TEXT, alpha=0.3, lw=1)
    ax_scatter.axvline(0, color=TEXT, alpha=0.3, lw=1)
    ax_scatter.set_title("surprise vs 21D forward return", color=TEXT, loc="left", fontsize=12)
    ax_scatter.set_xlabel("EPS surprise %")
    ax_scatter.set_ylabel("21D return %")

    ax_vix.plot(idx, vix, color=CYAN, lw=2.0)
    ax_vix.axhspan(25, 45, color=PINK, alpha=0.09)
    ax_vix.set_title("CBOE VIX context", color=TEXT, loc="left", fontsize=12)
    ax_vix.set_ylabel("index")
    save(fig, out)


def plot_sec_fundamental_price_mosaic(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(420)
    price = pd.Series(np.cumprod(1 + rng.normal(0.00048, 0.013, len(idx))) * 185, index=idx)
    volume_z = pd.Series(rng.normal(0, 0.65, len(idx)), index=idx).rolling(5).mean()
    event_ix = [58, 118, 181, 246, 318, 382]
    event_dates = idx[event_ix]
    labels = ["10-K", "10-Q", "8-K", "Form 4", "10-Q", "13F-HR"]
    ratio_names = ["P/E TTM", "Gross margin", "FCF margin", "ROE"]
    ratios = pd.Series([29.4, 46.2, 31.8, 148.0], index=ratio_names)
    statements = pd.DataFrame({
        "revenue": [394, 383, 391, 410, 428],
        "operating cash flow": [122, 111, 118, 126, 132],
        "free cash flow": [107, 99, 105, 113, 119],
    }, index=["FY21", "FY22", "FY23", "FY24", "FY25"])

    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
    dashboard_title(fig, human_title(title), "SEC filing dates, company facts, FMP fundamentals and adjusted price-volume reaction")
    ax_price = fig.add_subplot(gs[0, 0])
    ax_vol = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_ratio = fig.add_subplot(gs[0, 1])
    ax_stmt = fig.add_subplot(gs[1, 1])
    for ax in [ax_price, ax_vol, ax_ratio, ax_stmt]:
        style_ax(ax, "")

    ax_price.plot(idx, price, color=CYAN, lw=2.2, label="AAPL adjusted close")
    for event_date, label in zip(event_dates, labels):
        ax_price.axvline(event_date, color=PINK, lw=1.1, alpha=0.85)
        ax_price.text(event_date, price.max() * 0.985, label, color=TEXT, fontsize=8, rotation=90, va="top", ha="right")
    ax_price.set_title("price path with SEC events", color=TEXT, loc="left", fontsize=12)
    style_legend(ax_price)

    ax_vol.bar(idx[-220:], volume_z[-220:].fillna(0), color=[PINK if v > 1 else BLUE for v in volume_z[-220:].fillna(0)], alpha=0.65)
    ax_vol.axhline(0, color=TEXT, alpha=0.35, lw=1)
    ax_vol.set_title("volume reaction z-score", color=TEXT, loc="left", fontsize=12)

    ax_ratio.barh(ratios.index, ratios.values, color=[BLUE, CYAN, GREEN, PINK], alpha=0.92)
    ax_ratio.set_title("FMP TTM ratio snapshot", color=TEXT, loc="left", fontsize=12)

    for col, color in zip(statements.columns, [BLUE, CYAN, PINK]):
        ax_stmt.plot(statements.index, statements[col], color=color, lw=2.0, marker="o", label=col)
    ax_stmt.set_title("statement history", color=TEXT, loc="left", fontsize=12)
    ax_stmt.set_ylabel("$bn")
    style_legend(ax_stmt)
    save(fig, out)


def plot_macro_rates_equity_factor_panel(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=96, freq="ME")
    cpi = pd.Series(2.6 + np.sin(np.linspace(0, 10, len(idx))) * 0.95 + rng.normal(0, 0.09, len(idx)), index=idx)
    unrate = pd.Series(4.1 + np.cos(np.linspace(0, 8, len(idx))) * 0.45 + rng.normal(0, 0.05, len(idx)), index=idx)
    fed = pd.Series(1.2 + np.linspace(0, 3.7, len(idx)) + np.sin(np.linspace(0, 8, len(idx))) * 0.55, index=idx).clip(0, 5.7)
    ten_y = pd.Series(2.0 + np.linspace(0, 2.1, len(idx)) + np.sin(np.linspace(0, 7, len(idx))) * 0.45, index=idx)
    factors = pd.Series({"Mkt-RF": 8.2, "SMB": -1.4, "HML": 3.1, "RMW": 2.6, "CMA": 1.8, "MOM": 5.9})
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "XLI"]
    matrix = np.array([
        [0.18, 0.09, 0.21, 0.07],
        [0.11, 0.15, 0.24, 0.05],
        [0.27, 0.10, 0.35, 0.18],
        [0.08, 0.12, 0.04, 0.10],
        [0.16, 0.05, 0.11, -0.02],
        [0.06, 0.08, 0.02, 0.05],
        [-0.03, 0.11, -0.07, 0.16],
        [0.12, 0.13, 0.15, 0.08],
    ])

    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2)
    dashboard_title(fig, human_title(title), "FRED macro series, Fama-French factors and sector ETF returns aligned by regime")
    ax_macro = fig.add_subplot(gs[0, 0])
    ax_factor = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_curve = fig.add_subplot(gs[1, 1])
    for ax in [ax_macro, ax_factor, ax_heat, ax_curve]:
        style_ax(ax, "")

    ax_macro.plot(idx, cpi, color=PINK, lw=2.0, label="CPI YoY (FRED:CPIAUCSL)")
    ax_macro.plot(idx, unrate, color=AMBER, lw=2.0, label="Unemployment (FRED:UNRATE)")
    ax_macro.plot(idx, fed, color=CYAN, lw=2.0, label="Fed Funds (FRED:FEDFUNDS)")
    ax_macro.set_title("macro state variables", color=TEXT, loc="left", fontsize=12)
    ax_macro.set_ylabel("%")
    style_legend(ax_macro)

    ax_factor.bar(factors.index, factors.values, color=[GREEN if value > 0 else PINK for value in factors], alpha=0.92)
    ax_factor.axhline(0, color=TEXT, alpha=0.35, lw=1)
    ax_factor.set_title("Fama-French factor returns", color=TEXT, loc="left", fontsize=12)
    ax_factor.set_ylabel("% annualized")

    ax_heat.imshow(matrix, cmap=CORR_CMAP, vmin=-0.15, vmax=0.35, aspect="auto")
    ax_heat.grid(False)
    ax_heat.set_xticks(range(4), ["rates up\ninflation up", "rates up\ninflation down", "rates down\ninflation up", "rates down\ninflation down"])
    ax_heat.set_yticks(range(len(sectors)), sectors)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax_heat.text(j, i, f"{matrix[i, j]:.0%}", color=TEXT, ha="center", va="center", fontsize=7)
    ax_heat.set_title("sector return by macro regime", color=TEXT, loc="left", fontsize=12)

    ax_curve.plot(idx, ten_y - fed, color=BLUE, lw=2.1, label="10Y minus Fed Funds")
    ax_curve.plot(idx, cpi - 2.0, color=PINK, lw=1.8, label="CPI gap to 2%")
    ax_curve.axhline(0, color=TEXT, alpha=0.35, lw=1)
    ax_curve.set_title("policy and inflation gaps", color=TEXT, loc="left", fontsize=12)
    style_legend(ax_curve)
    save(fig, out)


def plot_crypto_macro_liquidity_panel(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(420)
    btc = pd.Series(np.cumprod(1 + rng.normal(0.00070, 0.026, len(idx))), index=idx)
    eth = pd.Series(np.cumprod(1 + rng.normal(0.00078, 0.031, len(idx))), index=idx)
    funding = pd.Series(rng.normal(0.011, 0.032, len(idx)), index=idx).rolling(7).mean()
    open_interest = pd.Series(18 + np.linspace(0, 9, len(idx)) + np.sin(np.linspace(0, 12, len(idx))) * 2.2 + rng.normal(0, 0.5, len(idx)), index=idx)
    vix = pd.Series(18 + np.sin(np.linspace(0, 12, len(idx))) * 5 + rng.normal(0, 1.2, len(idx)), index=idx).clip(10, 45)
    rates = pd.Series(1.8 + np.linspace(0, 2.6, len(idx)) + np.sin(np.linspace(0, 8, len(idx))) * 0.45, index=idx)
    corr = pd.DataFrame(
        [[1.00, 0.78, -0.31, -0.22], [0.78, 1.00, -0.24, -0.18], [-0.31, -0.24, 1.00, 0.42], [-0.22, -0.18, 0.42, 1.00]],
        index=["BTC", "ETH", "VIX", "10Y"],
        columns=["BTC", "ETH", "VIX", "10Y"],
    )

    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2)
    dashboard_title(fig, human_title(title), "Exchange spot, funding, open interest, volatility and rates in one crypto-macro panel")
    ax_spot = fig.add_subplot(gs[0, 0])
    ax_fund = fig.add_subplot(gs[0, 1])
    ax_corr = fig.add_subplot(gs[1, 0])
    ax_macro = fig.add_subplot(gs[1, 1])
    for ax in [ax_spot, ax_fund, ax_corr, ax_macro]:
        style_ax(ax, "")

    ax_spot.plot(idx, btc / btc.iloc[0] * 100, color=BLUE, lw=2.1, label="BTC/USDT spot")
    ax_spot.plot(idx, eth / eth.iloc[0] * 100, color=PINK, lw=2.1, label="ETH/USDT spot")
    ax_spot.set_title("spot performance", color=TEXT, loc="left", fontsize=12)
    ax_spot.set_ylabel("index = 100")
    style_legend(ax_spot)

    sample = funding.dropna().tail(140) * 100
    ax_fund.bar(sample.index, sample.values, color=[GREEN if value > 0 else RED for value in sample.values], alpha=0.7)
    ax_fund.axhline(0, color=TEXT, alpha=0.35, lw=1)
    ax_fund2 = ax_fund.twinx()
    ax_fund2.plot(open_interest.index[-140:], open_interest[-140:], color=CYAN, lw=1.8)
    ax_fund2.tick_params(colors=MUTED, labelsize=9)
    ax_fund2.spines["right"].set_color("#17376f")
    ax_fund.set_title("funding and open interest", color=TEXT, loc="left", fontsize=12)

    ax_corr.imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax_corr.grid(False)
    ax_corr.set_xticks(range(len(corr.columns)), corr.columns)
    ax_corr.set_yticks(range(len(corr.index)), corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax_corr.text(j, i, f"{corr.iloc[i, j]:.2f}", color=TEXT, ha="center", va="center", fontsize=10)
    ax_corr.set_title("252D cross-correlation", color=TEXT, loc="left", fontsize=12)

    ax_macro.plot(idx, vix, color=PINK, lw=2.0, label="CBOE VIX")
    ax_macro2 = ax_macro.twinx()
    ax_macro2.plot(idx, rates, color=CYAN, lw=1.8, label="10Y Treasury")
    ax_macro2.tick_params(colors=MUTED, labelsize=9)
    ax_macro2.spines["right"].set_color("#17376f")
    ax_macro.set_title("macro stress context", color=TEXT, loc="left", fontsize=12)
    save(fig, out)


def plot_news_filings_reaction_panel(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(320)
    price = pd.Series(np.cumprod(1 + rng.normal(0.00046, 0.013, len(idx))) * 190, index=idx)
    event_positions = [38, 74, 113, 158, 196, 239, 286]
    event_types = ["Tiingo news", "SEC 10-Q", "Finnhub news", "Form 4", "Earnings", "SEC 8-K", "Tiingo news"]
    events = pd.DataFrame({"date": idx[event_positions], "type": event_types})
    source_counts = events["type"].value_counts().sort_values()
    reactions = pd.Series({"Tiingo news": 0.006, "Finnhub news": 0.004, "SEC 10-Q": -0.012, "SEC 8-K": 0.009, "Form 4": -0.004, "Earnings": 0.018})
    peers = pd.DataFrame({
        "AAPL": np.cumprod(1 + rng.normal(0.00048, 0.013, len(idx))),
        "MSFT": np.cumprod(1 + rng.normal(0.00044, 0.012, len(idx))),
        "NVDA": np.cumprod(1 + rng.normal(0.00076, 0.023, len(idx))),
        "GOOGL": np.cumprod(1 + rng.normal(0.00040, 0.014, len(idx))),
    }, index=idx)

    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
    dashboard_title(fig, human_title(title), "News, filings, earnings and insider events aligned to adjusted price reaction")
    ax_price = fig.add_subplot(gs[0, 0])
    ax_peer = fig.add_subplot(gs[1, 0])
    ax_react = fig.add_subplot(gs[0, 1])
    ax_count = fig.add_subplot(gs[1, 1])
    for ax in [ax_price, ax_peer, ax_react, ax_count]:
        style_ax(ax, "")

    ax_price.plot(idx, price, color=CYAN, lw=2.2, label="AAPL adjusted close")
    color_map = {"Tiingo news": BLUE, "Finnhub news": CYAN, "SEC 10-Q": PINK, "SEC 8-K": PINK, "Form 4": AMBER, "Earnings": GREEN}
    for row in events.itertuples():
        ax_price.axvline(row.date, color=color_map.get(row.type, PINK), alpha=0.85, lw=1.15)
        ax_price.text(row.date, price.max() * 0.985, row.type, color=TEXT, fontsize=7.5, rotation=90, va="top", ha="right")
    ax_price.set_title("issuer event timeline", color=TEXT, loc="left", fontsize=12)
    style_legend(ax_price)

    for col, color in zip(peers.columns, [BLUE, CYAN, PINK, GREEN]):
        ax_peer.plot(peers.index, peers[col] / peers[col].iloc[0] * 100, color=color, lw=1.8, label=col)
    ax_peer.set_title("peer price context", color=TEXT, loc="left", fontsize=12)
    ax_peer.set_ylabel("index = 100")
    style_legend(ax_peer)

    ax_react.barh(reactions.index, reactions.values * 100, color=[GREEN if value > 0 else PINK for value in reactions], alpha=0.9)
    ax_react.axvline(0, color=TEXT, alpha=0.35, lw=1)
    ax_react.set_title("median 5D reaction by source", color=TEXT, loc="left", fontsize=12)
    ax_react.set_xlabel("%")

    ax_count.barh(source_counts.index, source_counts.values, color=[BLUE, CYAN, PINK, AMBER, GREEN][: len(source_counts)], alpha=0.9)
    ax_count.set_title("event objects joined", color=TEXT, loc="left", fontsize=12)
    save(fig, out)


def _single_ax(title: str, subtitle: str, figsize: tuple[float, float] = (11, 6)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor=NAVY)
    dashboard_title(fig, title, subtitle)
    style_ax(ax, "")
    return fig, ax


def _simple_series(seed: str, n: int = 360, drift: float = 0.00035, vol: float = 0.012) -> pd.Series:
    rng = rng_for(seed)
    idx = dates(n)
    return pd.Series(np.cumprod(1 + rng.normal(drift, vol, n)), index=idx)


def plot_visual_waterfall_attribution(stem: str, out: Path) -> None:
    title = "Waterfall Multi-Source Attribution"
    subtitle = "Pricing, factor, macro and cost effects decomposed into one attribution object"
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=NAVY)
    dashboard_title(fig, title, subtitle)
    style_ax(ax, "")

    components = pd.Series(
        {
            "Start": 2.5,
            "Sector\nallocation": 0.8,
            "Stock\nselection": 1.2,
            "FF Mkt-RF": -0.9,
            "FF SMB": 0.3,
            "Rates\nshock": -1.5,
            "Inflation\nshock": 0.4,
            "Costs": -0.2,
        }
    )
    starts = components.cumsum().shift(1).fillna(0)
    colors = [BLUE] + [GREEN if value >= 0 else RED for value in components.iloc[1:]]
    x = np.arange(len(components))
    ax.bar(x, components, bottom=starts, color=colors, alpha=0.92, width=0.66)
    ax.plot(x, starts + components, color=TEXT, alpha=0.45, linewidth=1.0)
    ax.axhline(0, color=TEXT, alpha=0.35, linewidth=1)
    for i, (base, value) in enumerate(zip(starts, components)):
        y = base + value / 2
        ax.text(i, y, f"{value:+.1f}", ha="center", va="center", color=TEXT, fontsize=8, weight="semibold")
    ax.set_xticks(x, components.index, rotation=0)
    ax.set_ylabel("contribution, percentage points")
    save(fig, out)


def plot_visual_radar_profiles(stem: str, out: Path) -> None:
    title = "Radar Multi-Metric Profiles"
    subtitle = "Peer names compared across valuation, quality, crowding, short pressure and macro/vol sensitivities"
    labels = ["P/E\ninverse", "ROE", "Gross\nmargin", "Crowding", "Days to\ncover", "Market\nbeta", "Rates\nsens", "VIX\nbeta"]
    aapl = np.array([0.72, 0.92, 0.84, 0.58, 0.38, 0.74, 0.33, 0.55])
    nvda = np.array([0.42, 0.78, 0.95, 0.88, 0.70, 0.96, 0.48, 0.86])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.r_[angles, angles[0]]
    aapl = np.r_[aapl, aapl[0]]
    nvda = np.r_[nvda, nvda[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True}, facecolor=NAVY)
    dashboard_title(fig, title, subtitle)
    ax.set_facecolor(NAVY)
    ax.plot(angles, aapl, color=CYAN, linewidth=2.2, marker="o", markersize=4, label="AAPL")
    ax.fill(angles, aapl, color=CYAN, alpha=0.18)
    ax.plot(angles, nvda, color=PINK, linewidth=2.2, marker="o", markersize=4, label="NVDA")
    ax.fill(angles, nvda, color=PINK, alpha=0.18)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1], labels)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color=MUTED, fontsize=8)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(True, color=GRID, alpha=0.55, linewidth=0.7)
    ax.spines["polar"].set_color("#17376f")
    style_legend(ax)
    save(fig, out)


def plot_visual_treemap_liquidity_crowding(stem: str, out: Path) -> None:
    title = "Treemap Liquidity and Crowding"
    subtitle = "Tile size is ADV proxy; color is ownership concentration and short-interest pressure"
    names = ["AAPL\nTech", "MSFT\nTech", "NVDA\nTech", "JPM\nFinancials", "XOM\nEnergy", "UNH\nHealth", "LLY\nHealth", "COST\nStaples"]
    adv = np.array([2800, 2100, 1450, 620, 520, 420, 360, 330], dtype=float)
    crowd = np.array([0.72, 0.68, 0.91, 0.45, 0.38, 0.52, 0.81, 0.57])
    short_z = np.array([0.2, -0.1, 1.4, 0.5, -0.8, 0.3, 0.9, -0.2])

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=NAVY)
    dashboard_title(fig, title, subtitle)
    style_ax(ax, "")
    ax.axis("off")

    total = adv.sum()
    x0 = 0.02
    y0 = 0.08
    width = 0.96
    height = 0.74
    running_x = x0
    cmap = plt.get_cmap("RdYlGn_r")
    for name, size, concentration, sz in zip(names, adv, crowd, short_z):
        rect_w = width * size / total
        color = cmap(np.clip(concentration, 0, 1))
        ax.add_patch(plt.Rectangle((running_x, y0), rect_w, height, facecolor=color, edgecolor=NAVY, linewidth=1.4))
        label_color = "#061226" if concentration < 0.75 else TEXT
        ax.text(running_x + rect_w / 2, y0 + height * 0.56, name, color=label_color, ha="center", va="center", fontsize=8, weight="semibold")
        ax.text(running_x + rect_w / 2, y0 + height * 0.42, f"ADV ${size:,.0f}m", color=label_color, ha="center", va="center", fontsize=7)
        ax.text(running_x + rect_w / 2, y0 + height * 0.31, f"crowd {concentration:.2f} | short z {sz:+.1f}", color=label_color, ha="center", va="center", fontsize=6.4)
        running_x += rect_w
    ax.text(0.02, 0.02, "left-to-right width = 63D dollar ADV proxy", transform=ax.transAxes, color=MUTED, fontsize=8)
    save(fig, out)


def plot_visual_parallel_coords(stem: str, out: Path) -> None:
    title = "Parallel Coordinates Screening"
    subtitle = "Universe screen across liquidity, valuation, quality, crowding, short pressure and macro sensitivities"
    tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "UNH", "LLY"]
    raw = pd.DataFrame(
        {
            "ADV $bn": [38, 31, 55, 8, 12, 6, 4],
            "P/E inv": [0.034, 0.029, 0.023, 0.120, 0.090, 0.065, 0.040],
            "ROE": [1.48, 0.36, 0.74, 0.15, 0.22, 0.28, 0.82],
            "Crowd z": [0.65, 0.72, 1.40, 0.30, -0.40, 0.55, 0.90],
            "Short z": [0.20, -0.10, 1.50, 0.60, -0.90, 0.10, 0.80],
            "Rates sens": [-0.60, -0.40, -0.20, 0.80, 0.30, -0.10, -0.50],
            "VIX beta": [0.80, 0.70, 1.20, 0.90, 0.50, 0.60, 0.85],
        },
        index=tickers,
    )
    norm = (raw - raw.min()) / (raw.max() - raw.min())
    score = norm["ADV $bn"] + norm["ROE"] + norm["P/E inv"] - norm["Crowd z"].clip(lower=0) * 0.35

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=NAVY)
    dashboard_title(fig, title, subtitle)
    style_ax(ax, "")
    x = np.arange(len(norm.columns))
    for ticker in norm.index:
        color = GREEN if score[ticker] > score.median() else BLUE
        if ticker == "NVDA":
            color = PINK
        ax.plot(x, norm.loc[ticker], marker="o", linewidth=2.0, markersize=4, alpha=0.88, color=color, label=ticker)
    ax.set_xticks(x, norm.columns, rotation=28, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("normalized metric")
    style_legend(ax)
    save(fig, out)


def plot_visual_conditioned_event_multiples(stem: str, out: Path) -> None:
    title = "Conditioned Event Small Multiples"
    subtitle = "Post-earnings return distributions conditioned by rates regime and liquidity bucket"
    rng = rng_for(stem)
    regimes = [("Rising rates", "High ADV", -0.1, BLUE), ("Rising rates", "Low ADV", -0.6, PINK), ("Falling rates", "High ADV", 0.8, CYAN), ("Falling rates", "Low ADV", 0.1, RED)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=NAVY, sharex=True, sharey=True)
    dashboard_title(fig, title, subtitle)
    for ax, (rates, liquidity, mu, color) in zip(axes.ravel(), regimes):
        style_ax(ax, "")
        values = rng.normal(mu, 2.35, 240)
        ax.hist(values, bins=26, color=color, alpha=0.86, edgecolor=NAVY, linewidth=0.7)
        median = np.median(values)
        ax.axvline(median, color=TEXT, linestyle="--", linewidth=1.1, alpha=0.85)
        ax.set_title(f"{rates} | {liquidity}", color=TEXT, fontsize=11, loc="left")
        ax.text(0.97, 0.90, f"median {median:+.1f}%", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("21D forward return after event, %")
    for ax in axes[:, 0]:
        ax.set_ylabel("event count")
    save(fig, out)


def plot_visual_sensitivity_heatmap(stem: str, out: Path) -> None:
    title = "Sensitivity Heatmap Clustered"
    subtitle = "Asset sensitivities to macro, factor and crowding proxies in a compact annotated matrix"
    assets = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "UNH"]
    factors = ["Rates\nDGS10", "Inflation\nCPI", "Growth\nPMI", "Mkt-RF\nFF", "Crowd z\n13F"]
    mat = np.array(
        [
            [-0.6, 0.2, 0.8, 1.1, 0.7],
            [-0.4, 0.1, 0.7, 0.9, 0.7],
            [-0.2, 0.3, 1.3, 1.4, 0.9],
            [0.8, -0.1, 0.4, 0.8, 0.5],
            [0.3, 0.6, 0.2, 0.6, -0.4],
            [-0.1, 0.1, 0.5, 0.8, 0.6],
        ]
    )
    row_order = [2, 0, 1, 5, 3, 4]
    mat = mat[row_order]
    assets = [assets[i] for i in row_order]
    fig, ax = plt.subplots(figsize=(10, 6.4), facecolor=NAVY)
    dashboard_title(fig, title, subtitle)
    style_ax(ax, "")
    im = ax.imshow(mat, cmap=CORR_CMAP, vmin=-1.5, vmax=1.5, aspect="auto")
    ax.grid(False)
    ax.set_xticks(np.arange(len(factors)), factors)
    ax.set_yticks(np.arange(len(assets)), assets)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", color=TEXT, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.025)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor("#17376f")
    save(fig, out)


def plot_88_plus_split(stem: str, panel: int, out: Path) -> None:
    title = human_title(stem[3:] if stem[:2].isdigit() else stem)
    lower = stem.lower()
    rng = rng_for(f"{stem}:{panel}")

    if lower.startswith("88_macro_positioning_cot_dashboard"):
        contracts, zscores, percentiles, weekly_change, states = cot_signal_data(stem)
        labels = ["ES", "CL", "GC", "ZN"]
        if panel == 1:
            fig, ax = _single_ax(title + " - Positioning Z-Score", "Managed-money net positioning z-score by CFTC contract")
            ax.bar(labels, zscores, color=[CYAN if x >= 0 else PINK for x in zscores], alpha=0.92)
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
            ax.set_ylabel("z-score")
        elif panel == 2:
            fig, ax = _single_ax(title + " - Positioning Percentile", "Current COT percentile using a three-year lookback")
            ax.bar(labels, percentiles, color=[PINK if x >= 80 or x <= 20 else CYAN for x in percentiles], alpha=0.92)
            ax.set_ylim(0, 100)
            ax.set_ylabel("percentile")
        elif panel == 3:
            fig, ax = _single_ax(title + " - Weekly Position Change", "One-week change in managed-money net contracts")
            ax.bar(labels, weekly_change / 1000, color=[GREEN if x >= 0 else RED for x in weekly_change], alpha=0.92)
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
            ax.set_ylabel("contracts, thousands")
        else:
            fig, ax = _single_ax(title + " - Signal State", "Positioning state with percentile and weekly change")
            ax.axis("off")
            for i, (code, state, pct, change) in enumerate(zip(labels, states, percentiles, weekly_change)):
                y = 0.80 - i * 0.18
                ax.text(0.08, y, code, color=MUTED, fontsize=10, transform=ax.transAxes)
                ax.text(0.20, y, state, color=PINK if "Crowded" in state else CYAN, fontsize=15, weight="semibold", transform=ax.transAxes)
                ax.text(0.20, y - 0.07, f"{pct:.0f}th percentile | {change:+,.0f} net contracts WoW", color=TEXT, fontsize=9, transform=ax.transAxes)
        save(fig, out)
        return

    if lower.startswith("89_usd_liquidity_cross_asset_mosaic"):
        idx = dates(520)
        walcl = pd.Series(8.7 + np.cumsum(rng.normal(-0.0007, 0.011, len(idx))), index=idx)
        tga = pd.Series(0.55 + np.sin(np.linspace(0, 11, len(idx))) * 0.18 + rng.normal(0, 0.025, len(idx)), index=idx)
        rrp = pd.Series(1.6 - np.linspace(0, 0.95, len(idx)) + np.sin(np.linspace(0, 7, len(idx))) * 0.22 + rng.normal(0, 0.025, len(idx)), index=idx).clip(0.05, None)
        liquidity = ((walcl.pct_change(26) - tga.pct_change(26) - rrp.pct_change(26)).rolling(8).mean() * 100).dropna()
        assets = pd.DataFrame({name: _simple_series(f"{stem}:{name}", 520, drift, vol) for name, drift, vol in [
            ("SPY equity", 0.00045, 0.011), ("TLT duration", 0.00008, 0.010), ("GLD gold", 0.00022, 0.009), ("UUP dollar", 0.00006, 0.005), ("BTC crypto", 0.00072, 0.026)
        ]})
        if panel == 1:
            fig, ax = _single_ax(title + " - USD Liquidity Impulse", "Fed balance sheet minus Treasury cash and reverse repo pressure")
            ax.plot(liquidity.index, liquidity, color=CYAN, lw=2.2)
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
            ax.fill_between(liquidity.index, 0, liquidity, where=liquidity >= 0, color=GREEN, alpha=0.12)
            ax.fill_between(liquidity.index, 0, liquidity, where=liquidity < 0, color=PINK, alpha=0.14)
            ax.set_ylabel("% impulse")
        elif panel == 2:
            fig, ax = _single_ax(title + " - Cross-Asset Price Paths", "Normalized SPY, TLT, GLD, UUP and BTC prices")
            for col, color in zip(assets.columns, [BLUE, CYAN, GREEN, AMBER, PINK]):
                ax.plot(assets.index, assets[col] / assets[col].iloc[0] * 100, color=color, lw=1.9, label=col)
            ax.set_ylabel("index = 100")
            style_legend(ax)
        elif panel == 3:
            fig, ax = _single_ax(title + " - Liquidity Components", "Fed balance sheet, Treasury account and reverse repo normalized")
            for series, label, color in [(walcl, "WALCL", BLUE), (tga, "WTREGEN", AMBER), (rrp, "RRPONTSYD", PINK)]:
                ax.plot(series.index, (series / series.iloc[0] - 1) * 100, color=color, lw=2.0, label=label)
            ax.set_ylabel("% since start")
            style_legend(ax)
        else:
            fig, ax = _single_ax(title + " - Liquidity Beta", "252D beta proxy to USD liquidity impulse")
            sens = pd.Series({"SPY": 0.42, "TLT": -0.18, "GLD": 0.16, "UUP": -0.31, "BTC": 0.58})
            ax.bar(sens.index, sens.values, color=[GREEN if x > 0 else PINK for x in sens], alpha=0.92)
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
        save(fig, out)
        return

    if lower.startswith("90_earnings_options_vol_reaction"):
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
        event_days = np.arange(-10, 22)
        implied = pd.Series([4.6, 4.1, 7.8, 5.2, 4.9, 5.5], index=symbols)
        realized = implied * pd.Series([0.82, 1.10, 1.34, 0.76, 0.91, 1.18], index=symbols)
        surprise = pd.Series([3.2, 1.6, 8.9, -0.8, 2.4, 4.1], index=symbols)
        fwd = pd.Series([2.1, 1.4, 6.8, -1.1, 0.9, 3.2], index=symbols)
        if panel == 1:
            fig, ax = _single_ax(title + " - Event Return Curve", "Average post-earnings price path around event day")
            y = pd.Series(np.tanh(event_days / 16) * 0.045 + rng.normal(0, 0.004, len(event_days)), index=event_days)
            ax.plot(event_days, y * 100, color=CYAN, lw=2.4)
            ax.axvline(0, color=TEXT, alpha=0.35, lw=1)
            ax.set_xlabel("trading days from earnings")
            ax.set_ylabel("% return")
        elif panel == 2:
            fig, ax = _single_ax(title + " - Implied vs Realized Move", "Option-implied earnings move compared with realized one-day move")
            x = np.arange(len(symbols))
            ax.bar(x - 0.18, implied, width=0.36, color=BLUE, alpha=0.9, label="implied move")
            ax.bar(x + 0.18, realized, width=0.36, color=PINK, alpha=0.9, label="realized move")
            ax.set_xticks(x, symbols)
            ax.set_ylabel("%")
            style_legend(ax)
        elif panel == 3:
            fig, ax = _single_ax(title + " - Surprise vs Forward Return", "EPS surprise percentage versus twenty-one-day forward return")
            ax.scatter(surprise, fwd, s=90, color=[BLUE, CYAN, PINK, GREEN, AMBER, RED], edgecolors=TEXT, linewidths=0.4)
            for symbol in symbols:
                ax.text(surprise[symbol] + 0.12, fwd[symbol], symbol, color=TEXT, fontsize=9)
            ax.axhline(0, color=TEXT, alpha=0.3, lw=1)
            ax.axvline(0, color=TEXT, alpha=0.3, lw=1)
            ax.set_xlabel("EPS surprise %")
            ax.set_ylabel("21D return %")
        else:
            fig, ax = _single_ax(title + " - VIX Context", "CBOE VIX around the event study window")
            idx = dates(260)
            vix = pd.Series(17 + np.sin(np.linspace(0, 11, len(idx))) * 5 + rng.normal(0, 1.2, len(idx)), index=idx).clip(10, 42)
            ax.plot(idx, vix, color=CYAN, lw=2.2)
            ax.axhspan(25, 45, color=PINK, alpha=0.10)
            ax.set_ylabel("VIX")
        save(fig, out)
        return

    if lower.startswith("91_sec_filing_fundamental_price_mosaic"):
        idx = dates(420)
        price = pd.Series(np.cumprod(1 + rng.normal(0.00048, 0.013, len(idx))) * 185, index=idx)
        events = idx[[58, 118, 181, 246, 318, 382]]
        labels = ["10-K", "10-Q", "8-K", "Form 4", "10-Q", "13F-HR"]
        if panel == 1:
            fig, ax = _single_ax(title + " - Price With SEC Events", "AAPL adjusted close with filing markers")
            ax.plot(idx, price, color=CYAN, lw=2.2)
            for date, label in zip(events, labels):
                ax.axvline(date, color=PINK, lw=1.1, alpha=0.85)
                ax.text(date, price.max() * 0.985, label, color=TEXT, fontsize=8, rotation=90, va="top", ha="right")
            ax.set_ylabel("price")
        elif panel == 2:
            fig, ax = _single_ax(title + " - Volume Reaction", "Five-day volume reaction z-score around filing periods")
            z = pd.Series(rng.normal(0, 0.65, len(idx)), index=idx).rolling(5).mean()
            ax.bar(idx[-220:], z[-220:].fillna(0), color=[PINK if v > 1 else BLUE for v in z[-220:].fillna(0)], alpha=0.7)
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
        elif panel == 3:
            fig, ax = _single_ax(title + " - Fundamental Snapshot", "FMP TTM ratio fields for the issuer")
            ratios = pd.Series({"P/E TTM": 29.4, "Gross margin": 46.2, "FCF margin": 31.8, "ROE": 148.0})
            ax.barh(ratios.index, ratios.values, color=[BLUE, CYAN, GREEN, PINK], alpha=0.92)
        else:
            fig, ax = _single_ax(title + " - Statement History", "Revenue, operating cash flow and free cash flow")
            statements = pd.DataFrame({"revenue": [394, 383, 391, 410, 428], "operating cash flow": [122, 111, 118, 126, 132], "free cash flow": [107, 99, 105, 113, 119]}, index=["FY21", "FY22", "FY23", "FY24", "FY25"])
            for col, color in zip(statements.columns, [BLUE, CYAN, PINK]):
                ax.plot(statements.index, statements[col], color=color, lw=2.2, marker="o", label=col)
            ax.set_ylabel("$bn")
            style_legend(ax)
        save(fig, out)
        return

    if lower.startswith("92_macro_rates_equity_factor_panel"):
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=96, freq="ME")
        cpi = pd.Series(2.6 + np.sin(np.linspace(0, 10, len(idx))) * 0.95 + rng.normal(0, 0.09, len(idx)), index=idx)
        unrate = pd.Series(4.1 + np.cos(np.linspace(0, 8, len(idx))) * 0.45 + rng.normal(0, 0.05, len(idx)), index=idx)
        fed = pd.Series(1.2 + np.linspace(0, 3.7, len(idx)) + np.sin(np.linspace(0, 8, len(idx))) * 0.55, index=idx).clip(0, 5.7)
        if panel == 1:
            fig, ax = _single_ax(title + " - Macro State Variables", "CPI, unemployment and Fed Funds from FRED-style macro feeds")
            ax.plot(idx, cpi, color=PINK, lw=2.1, label="CPI YoY")
            ax.plot(idx, unrate, color=AMBER, lw=2.1, label="Unemployment")
            ax.plot(idx, fed, color=CYAN, lw=2.1, label="Fed Funds")
            ax.set_ylabel("%")
            style_legend(ax)
        elif panel == 2:
            fig, ax = _single_ax(title + " - Fama-French Factor Returns", "Annualized factor returns aligned to the macro panel")
            factors = pd.Series({"Mkt-RF": 8.2, "SMB": -1.4, "HML": 3.1, "RMW": 2.6, "CMA": 1.8, "MOM": 5.9})
            ax.bar(factors.index, factors.values, color=[GREEN if x > 0 else PINK for x in factors], alpha=0.92)
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
            ax.set_ylabel("% annualized")
        elif panel == 3:
            fig, ax = _single_ax(title + " - Sector Return by Regime", "Sector ETFs conditioned by rates and inflation state")
            matrix = rng.normal(0.11, 0.11, (8, 4))
            ax.imshow(matrix, cmap=CORR_CMAP, vmin=-0.15, vmax=0.35, aspect="auto")
            ax.grid(False)
            ax.set_xticks(range(4), ["rates up\ninfl up", "rates up\ninfl down", "rates down\ninfl up", "rates down\ninfl down"])
            ax.set_yticks(range(8), ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "XLI"])
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, f"{matrix[i, j]:.0%}", color=TEXT, ha="center", va="center", fontsize=7)
        else:
            fig, ax = _single_ax(title + " - Policy and Inflation Gaps", "10Y minus Fed Funds and CPI gap to two percent")
            ten_y = pd.Series(2.0 + np.linspace(0, 2.1, len(idx)) + np.sin(np.linspace(0, 7, len(idx))) * 0.45, index=idx)
            ax.plot(idx, ten_y - fed, color=BLUE, lw=2.2, label="10Y minus Fed Funds")
            ax.plot(idx, cpi - 2.0, color=PINK, lw=1.9, label="CPI gap to 2%")
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
            style_legend(ax)
        save(fig, out)
        return

    if lower.startswith("93_crypto_macro_liquidity_panel"):
        idx = dates(420)
        btc = _simple_series(f"{stem}:btc", 420, 0.00070, 0.026)
        eth = _simple_series(f"{stem}:eth", 420, 0.00078, 0.031)
        if panel == 1:
            fig, ax = _single_ax(title + " - Spot Performance", "CCXT BTC/USDT and ETH/USDT normalized spot paths")
            ax.plot(idx, btc / btc.iloc[0] * 100, color=BLUE, lw=2.1, label="BTC/USDT")
            ax.plot(idx, eth / eth.iloc[0] * 100, color=PINK, lw=2.1, label="ETH/USDT")
            ax.set_ylabel("index = 100")
            style_legend(ax)
        elif panel == 2:
            fig, ax = _single_ax(title + " - Funding and Open Interest", "Perpetual funding pressure with open-interest context")
            funding = pd.Series(rng.normal(0.011, 0.032, len(idx)), index=idx).rolling(7).mean().tail(140) * 100
            oi = pd.Series(18 + np.linspace(0, 9, len(idx)) + np.sin(np.linspace(0, 12, len(idx))) * 2.2 + rng.normal(0, 0.5, len(idx)), index=idx).tail(140)
            ax.bar(funding.index, funding.values, color=[GREEN if x > 0 else RED for x in funding], alpha=0.7)
            ax2 = ax.twinx()
            ax2.plot(oi.index, oi.values, color=CYAN, lw=1.8)
            ax2.tick_params(colors=MUTED, labelsize=9)
            ax2.spines["right"].set_color("#17376f")
        elif panel == 3:
            fig, ax = _single_ax(title + " - Crypto Macro Correlation", "BTC and ETH versus VIX and 10Y rate proxies")
            corr = pd.DataFrame([[1, 0.78, -0.31, -0.22], [0.78, 1, -0.24, -0.18], [-0.31, -0.24, 1, 0.42], [-0.22, -0.18, 0.42, 1]], index=["BTC", "ETH", "VIX", "10Y"], columns=["BTC", "ETH", "VIX", "10Y"])
            ax.imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1)
            ax.grid(False)
            ax.set_xticks(range(4), corr.columns)
            ax.set_yticks(range(4), corr.index)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", color=TEXT, ha="center", va="center")
        else:
            fig, ax = _single_ax(title + " - Macro Stress Context", "CBOE VIX and 10Y Treasury rate context")
            vix = pd.Series(18 + np.sin(np.linspace(0, 12, len(idx))) * 5 + rng.normal(0, 1.2, len(idx)), index=idx).clip(10, 45)
            rates = pd.Series(1.8 + np.linspace(0, 2.6, len(idx)) + np.sin(np.linspace(0, 8, len(idx))) * 0.45, index=idx)
            ax.plot(idx, vix, color=PINK, lw=2.0, label="VIX")
            ax2 = ax.twinx()
            ax2.plot(idx, rates, color=CYAN, lw=1.8, label="10Y")
            ax2.tick_params(colors=MUTED, labelsize=9)
            ax2.spines["right"].set_color("#17376f")
        save(fig, out)
        return

    if lower.startswith("94_news_filings_price_reaction_panel"):
        idx = dates(320)
        price = pd.Series(np.cumprod(1 + rng.normal(0.00046, 0.013, len(idx))) * 190, index=idx)
        events = pd.DataFrame({"date": idx[[38, 74, 113, 158, 196, 239, 286]], "type": ["Tiingo news", "SEC 10-Q", "Finnhub news", "Form 4", "Earnings", "SEC 8-K", "Tiingo news"]})
        if panel == 1:
            fig, ax = _single_ax(title + " - Issuer Event Timeline", "News, filings, earnings and insider events over adjusted close")
            ax.plot(idx, price, color=CYAN, lw=2.2, label="AAPL adjusted close")
            for row in events.itertuples():
                ax.axvline(row.date, color=PINK, alpha=0.8, lw=1.1)
                ax.text(row.date, price.max() * 0.985, row.type, color=TEXT, fontsize=7.5, rotation=90, va="top", ha="right")
        elif panel == 2:
            fig, ax = _single_ax(title + " - Peer Price Context", "AAPL, MSFT, NVDA and GOOGL normalized price paths")
            for col, color, drift, vol in [("AAPL", BLUE, 0.00048, 0.013), ("MSFT", CYAN, 0.00044, 0.012), ("NVDA", PINK, 0.00076, 0.023), ("GOOGL", GREEN, 0.00040, 0.014)]:
                s = _simple_series(f"{stem}:{col}", 320, drift, vol)
                ax.plot(s.index, s / s.iloc[0] * 100, color=color, lw=1.8, label=col)
            ax.set_ylabel("index = 100")
            style_legend(ax)
        elif panel == 3:
            fig, ax = _single_ax(title + " - Median Five-Day Reaction", "Median AAPL price reaction by public event source")
            reactions = pd.Series({"Tiingo news": 0.006, "Finnhub news": 0.004, "SEC 10-Q": -0.012, "SEC 8-K": 0.009, "Form 4": -0.004, "Earnings": 0.018})
            ax.barh(reactions.index, reactions.values * 100, color=[GREEN if x > 0 else PINK for x in reactions], alpha=0.9)
            ax.axvline(0, color=TEXT, alpha=0.35, lw=1)
            ax.set_xlabel("%")
        else:
            fig, ax = _single_ax(title + " - Event Objects Joined", "Event counts by joined public source")
            counts = events["type"].value_counts().sort_values()
            ax.barh(counts.index, counts.values, color=[BLUE, CYAN, PINK, AMBER, GREEN][: len(counts)], alpha=0.9)
        save(fig, out)
        return

    if lower.startswith("95_spread_liquidity_monitoring_packet"):
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
        adv = pd.Series([38, 31, 55, 29, 26, 18], index=symbols)
        spread = pd.Series([2.8, 2.6, 5.9, 4.1, 3.4, 4.8], index=symbols)
        sl_adv = (spread / adv).rename("SL/ADV ratio")
        if panel == 1:
            fig, ax = _single_ax(title + " - Spread Proxy", "Estimated quoted spread proxy in basis points")
            ax.bar(symbols, spread, color=[BLUE, CYAN, PINK, GREEN, AMBER, RED], alpha=0.9)
            ax.set_ylabel("bps")
        elif panel == 2:
            fig, ax = _single_ax(title + " - SL/ADV Ratio", "Spread-liquidity ratio: spread proxy divided by dollar ADV rank")
            ax.bar(symbols, sl_adv * 100, color=[PINK if x > sl_adv.median() else CYAN for x in sl_adv], alpha=0.9)
            ax.set_ylabel("ratio x100")
        elif panel == 3:
            fig, ax = _single_ax(title + " - Days at 5% ADV", "Estimated days to execute a 25mm order at five percent ADV")
            days_trade = 25 / (adv * 0.05)
            ax.bar(symbols, days_trade, color=[PINK if x > 12 else CYAN for x in days_trade], alpha=0.9)
            ax.set_ylabel("days")
        else:
            fig, ax = _single_ax(title + " - Slippage Cost Curve", "Square-root slippage model by participation rate")
            participation = np.array([1, 2, 5, 10, 15, 20])
            for symbol, color in zip(["AAPL", "NVDA", "META"], [BLUE, PINK, CYAN]):
                ax.plot(participation, spread[symbol] * np.sqrt(participation / 5), color=color, lw=2.1, marker="o", label=symbol)
            ax.set_xlabel("participation rate % ADV")
            ax.set_ylabel("slippage proxy bps")
            style_legend(ax)
        save(fig, out)
        return

    if lower.startswith("96_earnings_revisions_price_momentum"):
        symbols = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "AMD"]
        breadth = pd.Series([0.82, 0.61, 0.48, 0.54, 0.51, 0.58, 0.72, 0.66], index=symbols)
        dispersion = pd.Series([0.21, 0.13, 0.10, 0.18, 0.12, 0.14, 0.20, 0.26], index=symbols)
        momentum = pd.Series([0.44, 0.19, 0.11, 0.16, 0.13, 0.21, 0.36, 0.29], index=symbols)
        accel = pd.Series([1.8, 0.7, 0.3, 0.5, 0.4, 0.6, 1.4, 1.1], index=symbols)
        if panel == 1:
            fig, ax = _single_ax(title + " - Revision Breadth", "Share of forward estimate fields revised upward")
            ax.bar(symbols, breadth, color=[GREEN if x > 0.6 else CYAN for x in breadth], alpha=0.9)
            ax.set_ylim(0, 1)
        elif panel == 2:
            fig, ax = _single_ax(title + " - Revision Dispersion", "Forward estimate dispersion across analyst rows")
            ax.bar(symbols, dispersion, color=[PINK if x > 0.2 else BLUE for x in dispersion], alpha=0.9)
        elif panel == 3:
            fig, ax = _single_ax(title + " - Revision Acceleration", "FY2 minus current-year forward estimate proxy")
            ax.bar(symbols, accel, color=[GREEN if x > 0 else PINK for x in accel], alpha=0.9)
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
        else:
            fig, ax = _single_ax(title + " - Revisions vs Price Momentum", "Revision breadth versus twelve-minus-one-month price momentum")
            ax.scatter(breadth, momentum, s=85, color=[BLUE, CYAN, PINK, GREEN, AMBER, RED, "#8ab4ff", "#b6e880"], edgecolors=TEXT, linewidths=0.4)
            for symbol in symbols:
                ax.text(breadth[symbol] + 0.01, momentum[symbol], symbol, color=TEXT, fontsize=8)
            ax.set_xlabel("revision breadth")
            ax.set_ylabel("12-1 momentum")
        save(fig, out)
        return

    if lower.startswith("97_roic_wacc_cash_conversion_quality"):
        symbols = ["MSFT", "AAPL", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "COST"]
        roic = pd.Series([0.31, 0.46, 0.52, 0.25, 0.29, 0.16, 0.34, 0.19], index=symbols)
        wacc = pd.Series([0.085, 0.083, 0.092, 0.086, 0.087, 0.090, 0.095, 0.079], index=symbols)
        fcf = pd.Series([0.92, 0.88, 0.79, 0.84, 0.96, 0.63, 0.74, 0.71], index=symbols)
        if panel == 1:
            fig, ax = _single_ax(title + " - ROIC vs WACC", "Return on invested capital versus cost of capital proxy")
            x = np.arange(len(symbols))
            ax.bar(x - 0.18, roic * 100, width=0.36, color=CYAN, alpha=0.9, label="ROIC")
            ax.bar(x + 0.18, wacc * 100, width=0.36, color=PINK, alpha=0.9, label="WACC proxy")
            ax.set_xticks(x, symbols)
            ax.set_ylabel("%")
            style_legend(ax)
        elif panel == 2:
            fig, ax = _single_ax(title + " - ROIC-WACC Spread", "Economic spread after subtracting WACC proxy")
            spread = roic - wacc
            ax.bar(symbols, spread * 100, color=[GREEN if x > 0 else PINK for x in spread], alpha=0.9)
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
            ax.set_ylabel("percentage points")
        elif panel == 3:
            fig, ax = _single_ax(title + " - Cash Conversion", "Free cash flow divided by net income proxy")
            ax.bar(symbols, fcf, color=[GREEN if x > 0.8 else CYAN for x in fcf], alpha=0.9)
            ax.axhline(1.0, color=TEXT, alpha=0.35, lw=1)
        else:
            fig, ax = _single_ax(title + " - Quality Score", "ROIC spread, cash conversion and balance sheet quality combined")
            score = (roic - wacc).rank(pct=True) + fcf.rank(pct=True)
            ax.barh(score.sort_values().index, score.sort_values().values, color=CYAN, alpha=0.9)
        save(fig, out)
        return

    if lower.startswith("98_oil_curve_regime_instrument_selection"):
        idx = dates(420)
        cl1 = pd.Series(78 + np.sin(np.linspace(0, 13, len(idx))) * 11 + rng.normal(0, 1.2, len(idx)), index=idx)
        spread = pd.Series(np.sin(np.linspace(0, 9, len(idx))) * 2.6 + rng.normal(0, 0.4, len(idx)), index=idx)
        if panel == 1:
            fig, ax = _single_ax(title + " - WTI Front-Month Reference", "CL1 WTI front-month reference price")
            ax.plot(idx, cl1, color=AMBER, lw=2.2)
            ax.set_ylabel("$ / bbl")
        elif panel == 2:
            fig, ax = _single_ax(title + " - Front-Second Spread", "CL1 minus CL2; positive values indicate backwardation")
            ax.plot(idx, spread, color=CYAN, lw=2.1)
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
            ax.fill_between(idx, 0, spread, where=spread > 0, color=GREEN, alpha=0.12)
            ax.fill_between(idx, 0, spread, where=spread < 0, color=PINK, alpha=0.12)
            ax.set_ylabel("$ spread")
        elif panel == 3:
            fig, ax = _single_ax(title + " - Instrument Oil Beta", "USO, XLE, XOP and SPY beta proxy to CL1 returns")
            beta = pd.Series({"USO": 0.82, "XLE": 0.54, "XOP": 0.68, "SPY": 0.19})
            ax.bar(beta.index, beta.values, color=[AMBER, CYAN, PINK, BLUE], alpha=0.9)
        else:
            fig, ax = _single_ax(title + " - COT Positioning Percentile", "Managed-money crude oil positioning state")
            vals = pd.Series({"net z-score": 0.88, "percentile": 68, "weekly change k": 9.2})
            ax.bar(vals.index, vals.values, color=[CYAN, PINK, GREEN], alpha=0.9)
        save(fig, out)
        return

    if lower.startswith("99_vixy_term_structure_macro_regime"):
        idx = dates(420)
        vix = pd.Series(18 + np.sin(np.linspace(0, 13, len(idx))) * 5.2 + rng.normal(0, 1.4, len(idx)), index=idx).clip(9, 55)
        slope = pd.Series(1.8 + np.sin(np.linspace(0, 12, len(idx))) * 2.1 + rng.normal(0, 0.35, len(idx)), index=idx)
        if panel == 1:
            fig, ax = _single_ax(title + " - VIX and VIXY", "Spot volatility context versus VIXY normalized price")
            ax.plot(idx, vix, color=CYAN, lw=2.0, label="VIX")
            ax2 = ax.twinx()
            vixy = _simple_series(f"{stem}:vixy", 420, -0.00025, 0.025)
            ax2.plot(vixy.index, vixy / vixy.iloc[0] * 100, color=PINK, lw=1.8)
            ax2.tick_params(colors=MUTED, labelsize=9)
            ax2.spines["right"].set_color("#17376f")
        elif panel == 2:
            fig, ax = _single_ax(title + " - VIX Term-Structure Slope", "Second-month minus front-month futures proxy")
            ax.plot(idx, slope, color=CYAN, lw=2.1)
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
            ax.fill_between(idx, 0, slope, where=slope > 0, color=PINK, alpha=0.12)
            ax.fill_between(idx, 0, slope, where=slope < 0, color=GREEN, alpha=0.12)
            ax.set_ylabel("vol points")
        elif panel == 3:
            fig, ax = _single_ax(title + " - Roll Carry Proxy", "Estimated monthly roll drag/benefit by term-structure state")
            carry = pd.Series({"contango drag": -6.4, "flat curve": -1.1, "backwardation": 3.8})
            ax.bar(carry.index, carry.values, color=[PINK, CYAN, GREEN], alpha=0.9)
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
            ax.set_ylabel("% monthly")
        else:
            fig, ax = _single_ax(title + " - Volatility Regime State", "VIX, VVIX and SKEW composite state")
            state = pd.Series({"VIX percentile": 0.62, "VVIX percentile": 0.71, "SKEW percentile": 0.58, "term slope pctile": 0.83})
            ax.barh(state.index, state.values, color=[BLUE, CYAN, PINK, AMBER], alpha=0.9)
            ax.set_xlim(0, 1)
        save(fig, out)
        return

    if lower.startswith("100_credit_hy_treasury_spread_regime"):
        idx = dates(420)
        hy = pd.Series(3.8 + np.sin(np.linspace(0, 12, len(idx))) * 1.1 + rng.normal(0, 0.10, len(idx)), index=idx).clip(2, 9)
        ig = pd.Series(1.2 + np.sin(np.linspace(0, 10, len(idx))) * 0.32 + rng.normal(0, 0.04, len(idx)), index=idx).clip(0.5, 3)
        if panel == 1:
            fig, ax = _single_ax(title + " - Credit Spreads", "High-yield and investment-grade option-adjusted spreads")
            ax.plot(idx, hy, color=PINK, lw=2.1, label="HY OAS")
            ax.plot(idx, ig, color=CYAN, lw=2.1, label="IG OAS")
            ax.set_ylabel("%")
            style_legend(ax)
        elif panel == 2:
            fig, ax = _single_ax(title + " - HY/IG Spread Ratio", "High-yield OAS divided by investment-grade OAS")
            ratio = hy / ig
            ax.plot(idx, ratio, color=AMBER, lw=2.2)
            ax.axhline(ratio.quantile(0.75), color=PINK, alpha=0.7, linestyle="--")
        elif panel == 3:
            fig, ax = _single_ax(title + " - Credit ETF Returns", "HYG, LQD, TLT and SPY sixty-three-day return proxy")
            vals = pd.Series({"HYG": 3.4, "LQD": 1.8, "TLT": -2.2, "SPY": 6.1})
            ax.bar(vals.index, vals.values, color=[CYAN, BLUE, PINK, GREEN], alpha=0.9)
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
            ax.set_ylabel("%")
        else:
            fig, ax = _single_ax(title + " - Credit-Rates Correlation", "ETF dependency surface across credit, duration and equity")
            corr = pd.DataFrame([[1, 0.72, -0.18, 0.64], [0.72, 1, 0.31, 0.42], [-0.18, 0.31, 1, -0.22], [0.64, 0.42, -0.22, 1]], index=["HYG", "LQD", "TLT", "SPY"], columns=["HYG", "LQD", "TLT", "SPY"])
            ax.imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1)
            ax.grid(False)
            ax.set_xticks(range(4), corr.columns)
            ax.set_yticks(range(4), corr.index)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", color=TEXT, ha="center", va="center")
        save(fig, out)
        return

    if lower.startswith("101_consumer_dispersion_not_the_trade"):
        names = ["WMT", "COST", "HD", "LOW", "MCD", "SBUX", "TSLA", "AMZN", "NKE"]
        returns_126 = pd.Series([0.08, 0.13, -0.02, -0.05, 0.04, -0.06, 0.19, 0.16, -0.11], index=names)
        margins = pd.Series([0.25, 0.13, 0.34, 0.33, 0.56, 0.29, 0.18, 0.48, 0.43], index=names)
        debt = pd.Series([0.72, 0.42, 3.1, 2.4, -6.2, 1.6, 0.18, 0.74, 0.91], index=names)
        if panel == 1:
            fig, ax = _single_ax(title + " - Staples vs Discretionary Spread", "XLP minus XLY one-hundred-twenty-six-day return spread")
            idx = dates(320)
            spread = pd.Series(np.sin(np.linspace(0, 11, len(idx))) * 0.10 + rng.normal(0, 0.012, len(idx)), index=idx)
            ax.plot(idx, spread * 100, color=CYAN, lw=2.2)
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
            ax.set_ylabel("return spread %")
        elif panel == 2:
            fig, ax = _single_ax(title + " - Consumer Equity Dispersion", "Six-month return by consumer-linked symbol")
            ax.bar(names, returns_126 * 100, color=[GREEN if x > 0 else PINK for x in returns_126], alpha=0.9)
            ax.axhline(0, color=TEXT, alpha=0.35, lw=1)
            ax.set_ylabel("%")
        elif panel == 3:
            fig, ax = _single_ax(title + " - Margin vs Leverage", "Gross margin and debt-to-equity dispersion")
            ax.scatter(margins, debt, s=90, color=[BLUE, CYAN, PINK, GREEN, AMBER, RED, "#8ab4ff", "#b6e880", "#c084fc"], edgecolors=TEXT, linewidths=0.4)
            for name in names:
                ax.text(margins[name] + 0.006, debt[name], name, color=TEXT, fontsize=8)
            ax.set_xlabel("gross margin TTM")
            ax.set_ylabel("debt / equity TTM")
        else:
            fig, ax = _single_ax(title + " - Consumer Resilience Score", "Return, margin, liquidity and leverage combined")
            score = returns_126.rank(pct=True) + margins.rank(pct=True) - debt.rank(pct=True)
            ax.barh(score.sort_values().index, score.sort_values().values, color=CYAN, alpha=0.9)
        save(fig, out)
        return

    plot_one(stem, out)


def plot_one(stem: str, out: Path) -> None:
    title = stem[3:] if stem[:2].isdigit() else stem
    lower = stem.lower()
    if lower.startswith("01_authentication_methods"):
        plot_01_authentication_methods(stem, out)
    elif lower.startswith("02_market_data_basics"):
        plot_02_market_data_basics(stem, out)
    elif lower.startswith("03_economic_data_macro"):
        plot_03_economic_data_macro(stem, out)
    elif lower.startswith("04_fundamental_analysis"):
        plot_04_fundamental_analysis(stem, out)
    elif lower.startswith("05_technical_analysis"):
        plot_05_technical_analysis(stem, out)
    elif lower.startswith("06_portfolio_analysis"):
        plot_06_portfolio_analysis(stem, out)
    elif lower.startswith("07_crypto_ccxt"):
        plot_07_crypto_ccxt(stem, out)
    elif lower.startswith("08_cboe_vix"):
        plot_08_cboe_vix(stem, out)
    elif lower.startswith("09_multpl_valuation"):
        plot_09_multpl_valuation(stem, out)
    elif lower.startswith("10_cftc_cot"):
        plot_10_cftc_cot(stem, out)
    elif "corporate_actions_pit_adjustments" in lower:
        plot_adjustment_semantics(stem, title, out)
    elif "domain_route_discovery_contract" in lower:
        plot_domain_contract(stem, title, out)
    elif "global_macro_sources" in lower:
        plot_macro_sources(stem, title, out)
    elif "options_vix_skew_term_structure" in lower:
        plot_vol_surface_context(stem, title, out)
    elif "index_constituents_universe_build" in lower:
        plot_universe_build(stem, title, out)
    elif "data_contract_lineage_audit" in lower:
        plot_investment_evidence_timeline(stem, title, out)
    elif "investment_evidence_packet" in lower:
        plot_investment_evidence_timeline(stem, title, out)
    elif "usd_liquidity_cross_asset_mosaic" in lower:
        plot_usd_liquidity_mosaic(stem, title, out)
    elif "earnings_options_vol_reaction" in lower:
        plot_earnings_options_reaction(stem, title, out)
    elif "sec_filing_fundamental_price_mosaic" in lower:
        plot_sec_fundamental_price_mosaic(stem, title, out)
    elif "macro_rates_equity_factor_panel" in lower:
        plot_macro_rates_equity_factor_panel(stem, title, out)
    elif "crypto_macro_liquidity_panel" in lower:
        plot_crypto_macro_liquidity_panel(stem, title, out)
    elif "news_filings_price_reaction_panel" in lower:
        plot_news_filings_reaction_panel(stem, title, out)
    elif "waterfall_multi_source_attribution" in lower:
        plot_visual_waterfall_attribution(stem, out)
    elif "radar_multi_metric_profiles" in lower:
        plot_visual_radar_profiles(stem, out)
    elif "treemap_liquidity_crowding" in lower:
        plot_visual_treemap_liquidity_crowding(stem, out)
    elif "parallel_coords_screening" in lower:
        plot_visual_parallel_coords(stem, out)
    elif "conditioned_event_small_multiples" in lower:
        plot_visual_conditioned_event_multiples(stem, out)
    elif "sensitivity_heatmap_clustered" in lower:
        plot_visual_sensitivity_heatmap(stem, out)
    elif "stress_testing_macro_scenarios" in lower or "macro_shock_cross_asset" in lower or "factor_macro_risk_shock_transmission" in lower:
        plot_scenario_bars(stem, title, out)
    elif "cta_futures_carry_trend_macro" in lower or "macro_positioning_cot" in lower:
        plot_positioning_dashboard(stem, title, out)
    elif "macro_regime_cot_cross_asset" in lower or "tactical_asset_allocation_macro_valuation" in lower or "macro_regime_allocation_control" in lower:
        plot_regime_timeline(stem, title, out)
    elif "inflation_shock_monitor" in lower:
        plot_inflation_monitor(stem, title, out)
    elif "institutional_crowding_13f_flows" in lower or "crowding_smart_money_flow" in lower:
        plot_flow_mosaic(stem, title, out)
    elif "crowding_liquidity_capacity_stress" in lower:
        plot_scatter(stem, title, out)
    elif "portfolio_morning_risk_brief" in lower or "integrated_daily_risk_attribution_report" in lower:
        plot_risk_contributors(stem, title, out)
    elif "hierarchical_risk_parity" in lower or "risk_parity" in lower:
        plot_weight_bar(stem, title, out)
    elif "market_data" in lower or "pricing" in lower:
        plot_multi_nav(stem, title, out)
    elif "macro" in lower or "economic" in lower or "rates" in lower:
        plot_yield_curve(stem, title, out)
    elif "fundamental" in lower or "valuation" in lower or "multpl" in lower:
        plot_bar(stem, title, out, ["AAPL", "MSFT", "GOOGL", "NVDA", "META"])
    elif "technical" in lower or "strategy_grid" in lower or "walk_forward" in lower:
        plot_grid(stem, title, out)
    elif "portfolio" in lower or "risk_parity" in lower or "allocation" in lower or "hrp" in lower:
        plot_bar(stem, title, out, ["SPY", "TLT", "GLD", "DBC", "UUP"])
    elif "crypto" in lower or "ccxt" in lower:
        plot_nav(stem, title, out)
    elif "monte_carlo" in lower:
        plot_monte_carlo(stem, title, out)
    elif "vix" in lower or "volatility" in lower or "tail_risk" in lower or "var_" in lower:
        plot_drawdown(stem, title, out)
    elif "cot" in lower or "positioning" in lower:
        plot_nav(stem, title, out)
    elif "correlation" in lower or "crowding" in lower or "risk_model" in lower:
        plot_heatmap(stem, title, out)
    elif "liquidity" in lower or "capacity" in lower or "universe" in lower:
        plot_scatter(stem, title, out)
    elif "event" in lower or "earnings" in lower or "pead" in lower or "congress" in lower:
        plot_event_curve(stem, title, out)
    elif "options" in lower or "greeks" in lower or "overlay" in lower:
        plot_payoff(stem, title, out)
    elif "drawdown" in lower or "performance" in lower:
        plot_drawdown(stem, title, out)
    elif "attribution" in lower or "factor" in lower or "brinson" in lower:
        plot_bar(stem, title, out, ["market", "size", "value", "quality", "momentum"])
    elif "auth" in lower:
        plot_bar(stem, title, out, ["API key", "JWT", "refresh", "scope"])
    else:
        plot_nav(stem, title, out)


def plot_outputs(stem: str, output_dir: Path) -> list[Path]:
    lower = stem.lower()
    try:
        number = int(stem.split("_", 1)[0])
    except ValueError:
        number = 0
    if lower.startswith("01_authentication_methods"):
        return []
    if lower.startswith("02_market_data_basics"):
        outputs = [
            output_dir / "02_market_01.png",
            output_dir / "02_market_02.png",
            output_dir / "02_market_03.png",
        ]
        plot_02_market_01(stem, outputs[0])
        plot_02_market_02(stem, outputs[1])
        plot_02_market_03(stem, outputs[2])
        return outputs
    if lower.startswith("03_economic_data_macro"):
        outputs = [
            output_dir / "03_macro_01_yield_curve.png",
            output_dir / "03_macro_02_inflation.png",
            output_dir / "03_macro_03_unemployment.png",
            output_dir / "03_macro_04_sensitivity.png",
        ]
        plot_03_macro_01(stem, outputs[0])
        plot_03_macro_02(stem, outputs[1])
        plot_03_macro_03(stem, outputs[2])
        plot_03_macro_04(stem, outputs[3])
        return outputs
    if lower.startswith("04_fundamental_analysis"):
        outputs = [
            output_dir / "04_fundamentals_01_valuation_quality.png",
            output_dir / "04_fundamentals_02_peer_pe.png",
            output_dir / "04_fundamentals_03_profitability.png",
        ]
        plot_04_fundamentals_01(stem, outputs[0])
        plot_04_fundamentals_02(stem, outputs[1])
        plot_04_fundamentals_03(stem, outputs[2])
        return outputs
    if lower.startswith("06_portfolio_analysis"):
        outputs = [
            output_dir / "06_portfolio_01_nav_risk.png",
            output_dir / "06_portfolio_02_risk_contribution.png",
            output_dir / "06_portfolio_03_correlation.png",
        ]
        plot_06_portfolio_01(stem, outputs[0])
        plot_06_portfolio_02(stem, outputs[1])
        plot_06_portfolio_03(stem, outputs[2])
        return outputs
    if lower.startswith("07_crypto_ccxt"):
        outputs = [
            output_dir / "07_crypto_01_spot_performance.png",
            output_dir / "07_crypto_02_funding_pressure.png",
            output_dir / "07_crypto_03_basis_state.png",
        ]
        plot_07_crypto_01(stem, outputs[0])
        plot_07_crypto_02(stem, outputs[1])
        plot_07_crypto_03(stem, outputs[2])
        return outputs
    if lower.startswith("08_cboe_vix"):
        outputs = [
            output_dir / "08_vix_01_vol_state.png",
            output_dir / "08_vix_02_drawdown_overlay.png",
            output_dir / "08_vix_03_regime_mix.png",
        ]
        plot_08_vix_01(stem, outputs[0])
        plot_08_vix_02(stem, outputs[1])
        plot_08_vix_03(stem, outputs[2])
        return outputs
    if lower.startswith("09_multpl_valuation"):
        outputs = [
            output_dir / "09_multpl_01_cape.png",
            output_dir / "09_multpl_02_rates.png",
            output_dir / "09_multpl_03_earnings_yield_spread.png",
        ]
        plot_09_multpl_01(stem, outputs[0])
        plot_09_multpl_02(stem, outputs[1])
        plot_09_multpl_03(stem, outputs[2])
        return outputs
    if lower.startswith("11_sec_filings"):
        outputs = [
            output_dir / "11_sec_01_filing_event_reaction.png",
        ]
        plot_11_sec_01(stem, outputs[0])
        return outputs
    if lower.startswith("12_finra_short_interest"):
        outputs = [
            output_dir / "12_finra_01_short_volume_ratio.png",
            output_dir / "12_finra_02_days_to_cover.png",
            output_dir / "12_finra_03_crowding_map.png",
        ]
        plot_12_finra_01(stem, outputs[0])
        plot_12_finra_02(stem, outputs[1])
        plot_12_finra_03(stem, outputs[2])
        return outputs
    if lower.startswith("13_openfigi"):
        outputs = [
            output_dir / "13_openfigi_01_aapl_identity.png",
        ]
        plot_13_openfigi_01(stem, outputs[0])
        return outputs
    if lower.startswith("14_corporate_actions_pit_adjustments"):
        outputs = [
            output_dir / "14_corporate_actions_01_raw_vs_adjusted.png",
        ]
        plot_14_actions_01(stem, outputs[0])
        return outputs
    if lower.startswith("15_domain_route_discovery_contract"):
        outputs = [
            output_dir / "15_domain_01_aapl_equity_packet.png",
        ]
        plot_15_domain_01(stem, outputs[0])
        return outputs
    if lower.startswith("16_global_macro_sources"):
        outputs = [
            output_dir / "16_macro_01_cpi_fed_funds.png",
            output_dir / "16_macro_02_global_growth.png",
            output_dir / "16_macro_03_energy_inflation.png",
        ]
        plot_16_sources_01(stem, outputs[0])
        plot_16_sources_02(stem, outputs[1])
        plot_16_sources_03(stem, outputs[2])
        return outputs
    if lower.startswith("17_options_vix_skew_term_structure"):
        outputs = [
            output_dir / "17_vol_01_vix_skew.png",
            output_dir / "17_vol_02_term_structure.png",
            output_dir / "17_vol_03_surface.png",
        ]
        plot_17_vol_01(stem, outputs[0])
        plot_17_vol_02(stem, outputs[1])
        plot_17_vol_03(stem, outputs[2])
        return outputs
    if lower.startswith("18_index_constituents_universe_build"):
        outputs = [
            output_dir / "18_index_01_universe_scores.png",
            output_dir / "18_index_02_sector_weights.png",
            output_dir / "18_index_03_liquidity_filter.png",
        ]
        plot_18_index_01(stem, outputs[0])
        plot_18_index_02(stem, outputs[1])
        plot_18_index_03(stem, outputs[2])
        return outputs
    if lower.startswith("19_data_contract_lineage_audit"):
        outputs = [
            output_dir / "19_lineage_01_aapl_evidence_timeline.png",
        ]
        plot_19_lineage_01(stem, outputs[0])
        return outputs
    if lower.startswith("90_waterfall_multi_source_attribution"):
        output = output_dir / f"{stem}_output_01.png"
        plot_visual_waterfall_attribution(stem, output)
        return [output]
    if lower.startswith("91_radar_multi_metric_profiles"):
        output = output_dir / f"{stem}_output_01.png"
        plot_visual_radar_profiles(stem, output)
        return [output]
    if lower.startswith("92_treemap_liquidity_crowding"):
        output = output_dir / f"{stem}_output_01.png"
        plot_visual_treemap_liquidity_crowding(stem, output)
        return [output]
    if lower.startswith("93_parallel_coords_screening"):
        output = output_dir / f"{stem}_output_01.png"
        plot_visual_parallel_coords(stem, output)
        return [output]
    if lower.startswith("94_conditioned_event_small_multiples"):
        output = output_dir / f"{stem}_output_01.png"
        plot_visual_conditioned_event_multiples(stem, output)
        return [output]
    if lower.startswith("95_sensitivity_heatmap_clustered"):
        output = output_dir / f"{stem}_output_01.png"
        plot_visual_sensitivity_heatmap(stem, output)
        return [output]
    if number >= 88:
        outputs = [output_dir / f"{stem}_{index}.png" for index in range(1, 5)]
        for index, output in enumerate(outputs, start=1):
            plot_88_plus_split(stem, index, output)
        return outputs
    output = output_dir / f"{stem}_output_01.png"
    plot_one(stem, output)
    return [output]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT), help="Directory for generated PNG files.")
    parser.add_argument("--output-style", choices=["preview", "run"], default="run", help="preview writes <stem>.png; run writes <stem>_output_01.png.")
    return parser.parse_args()


def output_name(stem: str, style: str) -> str:
    if style == "run":
        return f"{stem}_output_01.png"
    return f"{stem}.png"


def notebook_sort_key(path: Path) -> tuple[int, str]:
    prefix = path.stem.split("_", 1)[0]
    try:
        return int(prefix), path.stem
    except ValueError:
        return 10_000, path.stem


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.png"):
        old.unlink()
    notebooks = sorted(CANDIDATES.glob("*.ipynb"), key=notebook_sort_key)
    for notebook in notebooks:
        if args.output_style == "run":
            plot_outputs(notebook.stem, output_dir)
        else:
            output = output_dir / output_name(notebook.stem, args.output_style)
            if not notebook.stem.startswith("01_authentication_methods"):
                plot_one(notebook.stem, output)
    print(f"Generated {len(notebooks)} example charts in {output_dir}")


if __name__ == "__main__":
    main()
