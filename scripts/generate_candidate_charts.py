"""Generate one dark output chart for every example notebook in `_candidates/`.

These are visual outputs for the example catalog, not notebook execution
artifacts. The first notebook examples use denser institutional dashboard
layouts; every figure uses the same dark navy background.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
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
PINK = "#ff6fb3"
GREEN = "#66e0a3"
AMBER = "#ffc857"
RED = "#ff7a7a"
GRID = "#1f3a68"
CORR_CMAP = LinearSegmentedColormap.from_list("qj_corr", [CYAN, NAVY, PINK])


def rng_for(stem: str) -> np.random.Generator:
    seed = int(hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def style_ax(ax: plt.Axes, title: str) -> None:
    fig = ax.figure
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(PANEL)
    ax.set_title(title.replace("_", " "), color=TEXT, fontsize=12, pad=10, weight="medium", loc="left")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, alpha=0.45, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#17376f")
        spine.set_alpha(0.65)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def dashboard_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.patch.set_facecolor(NAVY)
    fig.text(0.035, 0.965, title, color=TEXT, fontsize=18, weight="semibold", va="top")
    fig.text(0.035, 0.925, subtitle, color=MUTED, fontsize=9, va="top")


def api_strip(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.035,
        0.035,
        text,
        color="#bcd4ff",
        fontsize=8.5,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#061641", "edgecolor": "#1a4a88", "alpha": 0.96},
    )


def style_legend(ax: plt.Axes) -> None:
    legend = ax.legend(facecolor=NAVY, edgecolor="#17376f", labelcolor=TEXT, fontsize=8)
    if legend:
        legend.get_frame().set_alpha(0.88)


def save(fig: plt.Figure, path: Path) -> None:
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.12, hspace=0.46, wspace=0.30)
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
    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], height_ratios=[1, 1])
    dashboard_title(fig, "Authentication and Scoped Access", "API key exchange, JWT TTL, route scopes and request audit context")
    ax_flow = fig.add_subplot(gs[:, 0])
    ax_scope = fig.add_subplot(gs[0, 1])
    ax_ttl = fig.add_subplot(gs[1, 1])
    for ax in [ax_flow, ax_scope, ax_ttl]:
        style_ax(ax, "")
    ax_flow.set_title("governed request path", color=TEXT, loc="left", fontsize=12)
    ax_flow.axis("off")
    steps = ["API key", "JWT access", "scope check", "route call", "request_id"]
    x = np.linspace(0.08, 0.92, len(steps))
    for i, (label, xpos) in enumerate(zip(steps, x)):
        ax_flow.scatter([xpos], [0.58], s=900, color=[BLUE, CYAN, PINK, GREEN, AMBER][i], alpha=0.95)
        ax_flow.text(xpos, 0.58, str(i + 1), color=NAVY, ha="center", va="center", fontsize=13, weight="bold")
        ax_flow.text(xpos, 0.38, label, color=TEXT, ha="center", va="center", fontsize=10)
        if i < len(steps) - 1:
            ax_flow.annotate("", xy=(x[i + 1] - 0.055, 0.58), xytext=(xpos + 0.055, 0.58), arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.8})
    scopes = ["pricing", "macro", "filings", "portfolio", "admin"]
    allowed = [1, 1, 1, 0.62, 0.18]
    ax_scope.barh(scopes, allowed, color=[GREEN, CYAN, BLUE, AMBER, RED], alpha=0.9)
    ax_scope.set_xlim(0, 1.05)
    ax_scope.set_title("tenant scope coverage", color=TEXT, loc="left", fontsize=12)
    ttl = pd.Series([15, 7 * 24 * 60], index=["access token", "refresh token"])
    ax_ttl.bar(ttl.index, ttl.values, color=[BLUE, PINK], alpha=0.92)
    ax_ttl.set_yscale("log")
    ax_ttl.set_ylabel("minutes, log scale")
    ax_ttl.set_title("token lifetime model", color=TEXT, loc="left", fontsize=12)
    api_strip(fig, "QuantJourneyAPI(api_key=...) -> JWT -> scoped route execution -> request_id + audit event")
    save(fig, out)


def plot_02_market_data_basics(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = dates(420)
    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 0.9])
    dashboard_title(fig, "Market Data: Adjusted OHLCV and Provider Evidence", "Normalized performance, liquidity, volatility regime and route metadata")
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
    ax_liq.set_title("ADV proxy", color=TEXT, loc="left", fontsize=12)
    ax_liq.set_ylabel("$bn")
    ax_meta.axis("off")
    meta = [("route", "qj.eod.get_historical_prices"), ("schema", "date/open/high/low/close/adj/volume"), ("policy", "adjusted=True"), ("lineage", "provider + request_id"), ("quality", "warnings[] retained")]
    for i, (k, v) in enumerate(meta):
        y = 0.88 - i * 0.16
        ax_meta.text(0.02, y, k.upper(), color=MUTED, fontsize=8, transform=ax_meta.transAxes)
        ax_meta.text(0.02, y - 0.055, v, color=TEXT, fontsize=9, transform=ax_meta.transAxes, family="monospace")
    api_strip(fig, "prices = qj.eod.get_historical_prices(symbol='AAPL', start_date=..., end_date=...)")
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
    ax_infl.plot(idx, cpi, color=PINK, lw=2.0, label="CPI YoY")
    ax_infl.axhline(2.0, color=GREEN, lw=1.4, linestyle="--", label="target")
    ax_infl.set_title("inflation vs target", color=TEXT, loc="left", fontsize=12)
    style_legend(ax_infl)
    unrate = pd.Series(3.9 + np.cos(np.linspace(0, 5, len(idx))) * 0.35 + rng.normal(0, 0.04, len(idx)), index=idx)
    ax_labor.plot(idx, unrate, color=AMBER, lw=2.0)
    ax_labor.set_title("unemployment regime", color=TEXT, loc="left", fontsize=12)
    ax_labor.set_ylabel("%")
    matrix = np.array([[0.72, 0.41, -0.18], [0.55, 0.63, 0.12], [-0.24, 0.31, 0.80]])
    ax_regime.imshow(matrix, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax_regime.set_xticks(range(3), ["growth", "inflation", "rates"])
    ax_regime.set_yticks(range(3), ["equity", "duration", "USD"])
    ax_regime.grid(False)
    ax_regime.set_title("macro sensitivity map", color=TEXT, loc="left", fontsize=12)
    api_strip(fig, "fred.get_cpi + fred.get_unemployment_rate + fred.get_treasury_10y -> aligned macro panel")
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
    ax_scatter.set_xlabel("P/E TTM")
    ax_scatter.set_ylabel("ROE")
    ax_scatter.set_title("valuation vs quality", color=TEXT, loc="left", fontsize=12)
    ax_bar.bar(peers, pe, color=colors, alpha=0.9)
    ax_bar.set_title("P/E TTM", color=TEXT, loc="left", fontsize=12)
    ax_margin.barh(peers, margin, color=colors, alpha=0.9)
    ax_margin.set_title("gross margin TTM", color=TEXT, loc="left", fontsize=12)
    api_strip(fig, "ratios = qj.fmp.get_financial_ratios_ttm(symbol='AAPL') -> peer-normalized valuation table")
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
    api_strip(fig, "prices -> SMA/RSI/MACD/Bollinger-style indicators -> chart-ready technical state")
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
    ax_nav.set_title("portfolio NAV", color=TEXT, loc="left", fontsize=12)
    ax_risk.bar(risk.index, risk.values, color=[BLUE, CYAN, PINK, GREEN, AMBER], alpha=0.9)
    ax_risk.set_title("risk contribution proxy", color=TEXT, loc="left", fontsize=12)
    ax_corr.imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax_corr.set_xticks(range(len(symbols)), symbols, rotation=35)
    ax_corr.set_yticks(range(len(symbols)), symbols)
    ax_corr.grid(False)
    ax_corr.set_title("correlation matrix", color=TEXT, loc="left", fontsize=12)
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            ax_corr.text(j, i, f"{corr.iloc[i, j]:.2f}", color=TEXT, ha="center", va="center", fontsize=8)
    api_strip(fig, "holdings + qj.eod.get_historical_prices(...) -> return, vol, drawdown, correlation, risk contribution")
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
    ax_nav.plot(idx, btc / btc.iloc[0] * 100, color=BLUE, lw=2.0, label="BTC/USDT")
    ax_nav.plot(idx, eth / eth.iloc[0] * 100, color=PINK, lw=2.0, label="ETH/USDT")
    ax_nav.set_title("normalized spot performance", color=TEXT, loc="left", fontsize=12)
    style_legend(ax_nav)
    ax_funding.bar(idx[-90:], funding[-90:] * 100, color=[GREEN if v > 0 else RED for v in funding[-90:]], alpha=0.72)
    ax_funding.set_title("funding pressure", color=TEXT, loc="left", fontsize=12)
    ax_funding.set_ylabel("bps")
    state = pd.Series({"spot": 0.86, "funding": 0.54, "open interest": 0.71, "basis": 0.63})
    ax_state.barh(state.index, state.values, color=[BLUE, CYAN, PINK, AMBER], alpha=0.9)
    ax_state.set_xlim(0, 1)
    ax_state.set_title("data coverage by surface", color=TEXT, loc="left", fontsize=12)
    api_strip(fig, "qj.ccxt.get_historical_prices + funding/open-interest feeds -> digital-asset exposure monitor")
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
    ax_vix.plot(idx, vix, color=CYAN, lw=2.0, label="VIX")
    ax_vix.plot(idx, vvix / 4, color=PINK, lw=1.8, label="VVIX / 4")
    ax_vix.axhspan(25, 55, color=RED, alpha=0.09)
    ax_vix.set_title("volatility state", color=TEXT, loc="left", fontsize=12)
    style_legend(ax_vix)
    ax_dd.fill_between(idx, spy_dd * 100, 0, color=PINK, alpha=0.22)
    ax_dd.plot(idx, spy_dd * 100, color=PINK, lw=1.4)
    ax_dd.set_title("drawdown overlay", color=TEXT, loc="left", fontsize=12)
    ax_dd.set_ylabel("%")
    regimes = pd.DataFrame({"calm": [0.18, 0.12], "normal": [0.55, 0.47], "elevated": [0.27, 0.41]}, index=["VIX", "VVIX"])
    ax_regime.imshow(regimes.values, cmap=LinearSegmentedColormap.from_list("vol", [BLUE, NAVY, RED]), vmin=0, vmax=0.6)
    ax_regime.set_xticks(range(3), regimes.columns)
    ax_regime.set_yticks(range(2), regimes.index)
    ax_regime.grid(False)
    ax_regime.set_title("regime distribution", color=TEXT, loc="left", fontsize=12)
    api_strip(fig, "qj.cboe.get_vix_data + get_vvix_data + get_skew_index_data -> risk regime diagnostics")
    save(fig, out)


def plot_09_multpl_valuation(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=180, freq="ME")
    cape = pd.Series(24 + np.linspace(0, 9, len(idx)) + 2.5 * np.sin(np.linspace(0, 10, len(idx))) + rng.normal(0, 0.7, len(idx)), index=idx)
    ten_y = pd.Series(2.0 + np.linspace(0, 2.3, len(idx)) + 0.7 * np.sin(np.linspace(0, 8, len(idx))), index=idx)
    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2)
    dashboard_title(fig, "Market Valuation: Shiller P/E, Rates and Percentile Bands", "Long-horizon valuation context joined to rates for asset-allocation review")
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
    metrics = pd.Series({"CAPE": 0.84, "dividend yield": 0.22, "earnings yield": 0.31, "rates": 0.76})
    ax_pct.barh(metrics.index, metrics.values, color=[PINK, GREEN, AMBER, CYAN], alpha=0.9)
    ax_pct.set_xlim(0, 1)
    ax_pct.set_title("historical percentile", color=TEXT, loc="left", fontsize=12)
    api_strip(fig, "qj.multpl.get_shiller_pe_ratio + rates -> valuation percentile packet")
    save(fig, out)


def plot_10_cftc_cot(stem: str, out: Path) -> None:
    rng = rng_for(stem)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=180, freq="W")
    net = pd.Series(np.sin(np.linspace(0, 12, len(idx))) * 1.2 + rng.normal(0, 0.22, len(idx)), index=idx)
    price = pd.Series(np.cumprod(1 + rng.normal(0.0009, 0.018, len(idx))) * 100, index=idx)
    fig = plt.figure(figsize=(13, 7), facecolor=NAVY)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1])
    dashboard_title(fig, "CFTC COT: Positioning, Crowding and Price Confirmation", "Futures positioning converted into z-score and signal state for macro/CTA research")
    ax_pos = fig.add_subplot(gs[0, :])
    ax_price = fig.add_subplot(gs[1, 0])
    ax_book = fig.add_subplot(gs[1, 1])
    for ax in [ax_pos, ax_price, ax_book]:
        style_ax(ax, "")
    ax_pos.bar(idx, net, width=5, color=[CYAN if v >= 0 else PINK for v in net], alpha=0.8)
    ax_pos.axhline(0, color=TEXT, alpha=0.45, lw=1)
    ax_pos.axhline(1, color=RED, alpha=0.5, linestyle="--")
    ax_pos.axhline(-1, color=GREEN, alpha=0.5, linestyle="--")
    ax_pos.set_title("managed-money net positioning z-score", color=TEXT, loc="left", fontsize=12)
    ax_price.plot(idx, price, color=AMBER, lw=2.0)
    ax_price.set_title("linked futures proxy", color=TEXT, loc="left", fontsize=12)
    book = pd.Series({"equity": 0.44, "rates": -0.31, "gold": 0.52, "oil": 0.18, "USD": -0.22})
    ax_book.barh(book.index, book.values, color=[CYAN if v > 0 else PINK for v in book], alpha=0.9)
    ax_book.axvline(0, color=TEXT, alpha=0.35, lw=1)
    ax_book.set_title("macro positioning mosaic", color=TEXT, loc="left", fontsize=12)
    api_strip(fig, "qj.cftc.get_cot_summary + futures pricing -> positioning dashboard")
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
        ("AAPL", BLUE, 0.00055, 0.014),
        ("MSFT", CYAN, 0.00050, 0.012),
        ("NVDA", PINK, 0.00078, 0.021),
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
    ax.set_ylabel("score")
    save(fig, out)


def plot_heatmap(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    labels = ["SPY", "TLT", "GLD", "UUP", "BTC"]
    x = rng.normal(size=(420, len(labels)))
    corr = np.corrcoef(x.T)
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    style_ax(ax, title)
    ax.grid(False)
    ax.imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
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
    ax.set_ylabel("drawdown (%)")
    save(fig, out)


def plot_yield_curve(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    tenors = ["3M", "1Y", "2Y", "5Y", "10Y", "30Y"]
    base = np.array([4.95, 4.75, 4.55, 4.38, 4.30, 4.42])
    curve = base + rng.normal(0, 0.05, len(base))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(tenors, curve, color=CYAN, linewidth=2.8, marker="o", markersize=7)
    ax.set_ylabel("yield (%)")
    save(fig, out)


def plot_payoff(stem: str, title: str, out: Path) -> None:
    spot = np.linspace(70, 130, 121)
    payoff = np.minimum(np.maximum(spot - 100, -12), 16) + 2.5
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(spot, payoff, color=GREEN, linewidth=2.5)
    ax.axhline(0, color=TEXT, alpha=0.35, linewidth=1.0)
    ax.set_xlabel("underlying at expiry")
    ax.set_ylabel("overlay payoff")
    save(fig, out)


def plot_scatter(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    x = rng.lognormal(18, 0.7, 38) / 1e6
    y = rng.normal(0.18, 0.07, 38)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.scatter(x, y, s=58, color=CYAN, alpha=0.82, edgecolors=TEXT, linewidths=0.35)
    ax.set_xscale("log")
    ax.set_xlabel("ADV / capacity proxy")
    ax.set_ylabel("risk or score")
    save(fig, out)


def plot_event_curve(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    x = np.arange(-10, 43)
    y = np.tanh(x / 18) * 0.05 + rng.normal(0, 0.004, len(x))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(x, y * 100, color=AMBER, linewidth=2.4)
    ax.axvline(0, color=TEXT, alpha=0.35, linewidth=1.0)
    ax.set_xlabel("days from event")
    ax.set_ylabel("avg return (%)")
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
    ax.set_ylabel("simulated NAV")
    save(fig, out)


def plot_grid(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    z = rng.normal(0.9, 0.35, (7, 7))
    z += np.linspace(-0.2, 0.3, 7)[None, :]
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    style_ax(ax, title)
    ax.grid(False)
    ax.imshow(z, cmap="magma")
    ax.set_xlabel("slow window")
    ax.set_ylabel("fast window")
    save(fig, out)


def plot_evidence_packet(stem: str, title: str, out: Path) -> None:
    labels = ["price", "ratios", "filings", "insiders", "identity", "shorts", "vol"]
    values = [0.92, 0.78, 0.64, 0.48, 0.86, 0.52, 0.71]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title, )
    ax.bar(labels, values, color=[BLUE, CYAN, PINK, GREEN, AMBER, RED, "#8ab4ff"], alpha=0.92)
    ax.set_ylabel("evidence coverage")
    ax.set_ylim(0, 1.05)
    save(fig, out)


def plot_scenario_bars(stem: str, title: str, out: Path) -> None:
    labels = ["inflation", "hard landing", "USD squeeze", "risk-on"]
    values = [-3.8, -5.6, -2.9, 4.4]
    colors = [RED if value < 0 else GREEN for value in values]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.bar(labels, values, color=colors, alpha=0.9)
    ax.axhline(0, color=TEXT, alpha=0.35, linewidth=1)
    ax.set_ylabel("portfolio impact (%)")
    save(fig, out)


def plot_regime_timeline(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    x = dates(420)
    growth = np.sin(np.linspace(0, 10, len(x))) + rng.normal(0, 0.12, len(x))
    inflation = np.cos(np.linspace(0, 8, len(x))) + rng.normal(0, 0.12, len(x))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(x, growth, color=CYAN, linewidth=2.1, label="growth composite")
    ax.plot(x, inflation, color=PINK, linewidth=2.1, label="inflation pressure")
    ax.fill_between(x, -2, 2, where=(growth < 0), color=RED, alpha=0.08)
    ax.fill_between(x, -2, 2, where=(inflation > 0), color=AMBER, alpha=0.08)
    ax.legend(facecolor=PANEL, edgecolor="#17376f", labelcolor=TEXT)
    ax.set_ylabel("regime score")
    save(fig, out)


def plot_inflation_monitor(stem: str, title: str, out: Path) -> None:
    labels = ["XLE", "XLK", "XLF", "XLU", "XLP", "XLY"]
    values = [0.82, -0.21, 0.18, -0.33, -0.08, 0.27]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.bar(labels, values, color=[GREEN if value > 0 else PINK for value in values], alpha=0.92)
    ax.axhline(0, color=TEXT, alpha=0.35, linewidth=1)
    ax.set_ylabel("inflation beta proxy")
    save(fig, out)


def plot_positioning_dashboard(stem: str, title: str, out: Path) -> None:
    labels = ["SPX", "gold", "crude", "USD", "10Y"]
    values = [1.4, -0.7, 0.9, 1.1, -1.2]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.bar(labels, values, color=[CYAN if value >= 0 else PINK for value in values], alpha=0.92)
    ax.axhline(0, color=TEXT, alpha=0.35, linewidth=1)
    ax.set_ylabel("positioning z-score")
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
    ax.plot(close.index, close, color=CYAN, linewidth=2.1, label="close")
    ax.plot(adjusted.index, adjusted, color=PINK, linewidth=2.1, label="adjusted close")
    ax.legend(facecolor=PANEL, edgecolor="#17376f", labelcolor=TEXT)
    ax.set_ylabel("price")
    save(fig, out)


def plot_domain_contract(stem: str, title: str, out: Path) -> None:
    labels = ["domains", "aliases", "described", "scoped", "callable"]
    values = [540, 120, 5, 5, 4]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.bar(labels, values, color=[BLUE, CYAN, PINK, GREEN, AMBER], alpha=0.92)
    ax.set_ylabel("contract objects")
    save(fig, out)


def plot_macro_sources(stem: str, title: str, out: Path) -> None:
    labels = ["FRED", "IMF", "OECD", "WorldBank", "DBnomics", "Eurostat"]
    values = [1.0, 0.82, 0.74, 0.86, 0.68, 0.59]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.bar(labels, values, color=[BLUE, CYAN, PINK, GREEN, AMBER, RED], alpha=0.92)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("coverage score")
    save(fig, out)


def plot_vol_surface_context(stem: str, title: str, out: Path) -> None:
    rng = rng_for(stem)
    x = dates(360)
    vix = pd.Series(18 + 5 * np.sin(np.linspace(0, 13, len(x))) + rng.normal(0, 1.5, len(x)), index=x).clip(9, 55)
    skew = pd.Series(125 + 9 * np.cos(np.linspace(0, 10, len(x))) + rng.normal(0, 3, len(x)), index=x)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.plot(vix.index, vix, color=CYAN, linewidth=2.1, label="VIX")
    ax.plot(skew.index, (skew - 100) / 2, color=PINK, linewidth=2.1, label="SKEW scaled")
    ax.legend(facecolor=PANEL, edgecolor="#17376f", labelcolor=TEXT)
    ax.set_ylabel("vol context")
    save(fig, out)


def plot_universe_build(stem: str, title: str, out: Path) -> None:
    labels = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM"]
    values = [0.72, 0.69, 0.84, 0.58, 0.61, 0.64, 0.47, 0.43]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_ax(ax, title)
    ax.bar(labels, values, color=[BLUE, CYAN, PINK, GREEN, AMBER, RED, "#8ab4ff", "#b6e880"], alpha=0.92)
    ax.set_ylim(0, 1)
    ax.set_ylabel("universe score")
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
        plot_lineage_packet(stem, title, out)
    elif "investment_evidence_packet" in lower:
        plot_evidence_packet(stem, title, out)
    elif "macro_shock_cross_asset" in lower:
        plot_scenario_bars(stem, title, out)
    elif "macro_regime_allocation_control" in lower:
        plot_regime_timeline(stem, title, out)
    elif "inflation_shock_monitor" in lower:
        plot_inflation_monitor(stem, title, out)
    elif "macro_positioning_cot" in lower:
        plot_positioning_dashboard(stem, title, out)
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
    elif "monte_carlo" in lower:
        plot_monte_carlo(stem, title, out)
    elif "drawdown" in lower or "performance" in lower:
        plot_drawdown(stem, title, out)
    elif "attribution" in lower or "factor" in lower or "brinson" in lower:
        plot_bar(stem, title, out, ["market", "size", "value", "quality", "momentum"])
    elif "auth" in lower:
        plot_bar(stem, title, out, ["API key", "JWT", "refresh", "scope"])
    else:
        plot_nav(stem, title, out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT), help="Directory for generated PNG files.")
    parser.add_argument("--output-style", choices=["preview", "run"], default="run", help="preview writes <stem>.png; run writes <stem>_output_01.png.")
    return parser.parse_args()


def output_name(stem: str, style: str) -> str:
    if style == "run":
        return f"{stem}_output_01.png"
    return f"{stem}.png"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.png"):
        old.unlink()
    notebooks = sorted(CANDIDATES.glob("*.ipynb"))
    for notebook in notebooks:
        plot_one(notebook.stem, output_dir / output_name(notebook.stem, args.output_style))
    print(f"Generated {len(notebooks)} example charts in {output_dir}")


if __name__ == "__main__":
    main()
