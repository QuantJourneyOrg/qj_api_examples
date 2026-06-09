"""Generate additional metric-rich showcase charts into `_output/_v3`.

The V3 pack is intentionally separate from the notebook output pipeline. It
adds finance-facing visual patterns that are useful for landing pages and sales
materials: quarterly statement bars, ratio-over-ratio panels and rolling
z-score diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

MPLCONFIG = Path(tempfile.gettempdir()) / "qj_api_examples_mplconfig"
MPLCONFIG.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "_output" / "_v3"

NAVY = "#020617"
PANEL = "#020617"
TEXT = "#edf4ff"
MUTED = "#a6b4d3"
GRID = "#1f3a68"
BLUE = "#2580d8"
CYAN = "#58d5ff"
ORANGE = "#ffb000"
RED = "#ff3b30"
PINK = "#c04486"
GREEN = "#66e0a3"
AMBER = "#f6b44b"
INK = "#061226"
CMAP = LinearSegmentedColormap.from_list("qj_v3", [CYAN, NAVY, PINK])


def quarter_labels(start_year: int, start_quarter: int, count: int) -> list[str]:
    labels = []
    year = start_year
    quarter = start_quarter
    for _ in range(count):
        labels.append(f"Q{quarter} '{str(year)[-2:]}")
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1
    return labels


def rolling_zscore(series: pd.Series, window: int = 8) -> pd.Series:
    mean = series.rolling(window, min_periods=max(4, window // 2)).mean()
    std = series.rolling(window, min_periods=max(4, window // 2)).std()
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def cagr(values: np.ndarray, periods_per_year: float = 4.0) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or values[0] <= 0 or values[-1] <= 0:
        return float("nan")
    years = (len(values) - 1) / periods_per_year
    return (values[-1] / values[0]) ** (1 / years) - 1


def total_change(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or values[0] == 0:
        return float("nan")
    return values[-1] / values[0] - 1


def fmt(value: float, suffix: str = "") -> str:
    if not np.isfinite(value):
        return "-"
    abs_value = abs(float(value))
    sign = "-" if value < 0 else ""
    if abs_value >= 1000:
        return f"{sign}{abs_value:,.0f}{suffix}"
    if abs_value >= 100:
        return f"{sign}{abs_value:.0f}{suffix}"
    if abs_value >= 10:
        return f"{sign}{abs_value:.1f}{suffix}"
    return f"{sign}{abs_value:.2f}{suffix}"


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, alpha=0.45, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#17376f")
        spine.set_alpha(0.55)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def style_3d(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color(MUTED)
        axis._axinfo["grid"]["color"] = "#1f3a68"
        axis._axinfo["grid"]["linewidth"] = 0.7
        axis._axinfo["grid"]["alpha"] = 0.45
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_facecolor(NAVY)
        pane.set_edgecolor("#17376f")
        pane.set_alpha(1.0)


def title(fig: plt.Figure, heading: str, subtitle: str) -> None:
    fig.patch.set_facecolor(NAVY)
    fig.text(0.035, 0.965, heading, color=TEXT, fontsize=22, weight="semibold", va="top")
    fig.text(0.035, 0.915, subtitle, color=MUTED, fontsize=10.5, va="top")


def surface_layout(fig: plt.Figure) -> tuple[plt.Axes, plt.Axes]:
    ax = fig.add_axes([0.08, 0.13, 0.74, 0.69], projection="3d")
    cax = fig.add_axes([0.865, 0.25, 0.020, 0.44])
    return ax, cax


def legend_below(ax: plt.Axes, columns: int = 1) -> None:
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.16),
        ncol=columns,
        frameon=False,
        labelcolor=TEXT,
        fontsize=9,
        handlelength=1.2,
    )
    if legend:
        for text in legend.get_texts():
            text.set_color(TEXT)


def annotate_bars(ax: plt.Axes, containers: list, fontsize: float = 8.5, as_pct: bool = False) -> None:
    for container in containers:
        labels = []
        for patch in container:
            h = patch.get_height()
            labels.append(f"{h:.0f}%" if as_pct else f"{h:,.0f}")
        ax.bar_label(container, labels=labels, padding=3, color=TEXT, fontsize=fontsize)


def annotate_sparse_bars(ax: plt.Axes, bars, values: np.ndarray, *, step: int, fontsize: float = 7.2) -> None:
    for index, (patch, value) in enumerate(zip(bars, values)):
        if index % step != 0 and index != len(values) - 1:
            continue
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        ax.text(x, y + (0.025 * max(values) if y >= 0 else -0.025 * max(abs(values))), fmt(float(y)), ha="center", va="bottom" if y >= 0 else "top", color=TEXT, fontsize=fontsize)


def sampled_ticks(labels: list[str], max_ticks: int = 18) -> tuple[list[int], list[str]]:
    if len(labels) <= max_ticks:
        idx = list(range(len(labels)))
    else:
        step = max(1, math.ceil(len(labels) / max_ticks))
        idx = list(range(0, len(labels), step))
        if idx[-1] != len(labels) - 1:
            idx.append(len(labels) - 1)
    return idx, [labels[i] for i in idx]


def save(fig: plt.Figure, out: Path) -> None:
    fig.subplots_adjust(left=0.065, right=0.965, top=0.82, bottom=0.17, hspace=0.46, wspace=0.30)
    fig.savefig(out, dpi=160, facecolor=NAVY, edgecolor="none")
    plt.close(fig)


def plot_cloud_arr(out: Path) -> None:
    q = quarter_labels(2022, 3, 15)
    gcp = np.array([27, 29, 30, 32, 34, 37, 38, 41, 45, 48, 49, 54, 61, 71, 80])
    azure = np.array([60, 64, 65, 71, 72, 78, 81, 88, 89, 95, 99, 112, 115, 123, 130])
    aws = np.array([82, 86, 85, 89, 92, 97, 100, 105, 110, 115, 117, 123, 132, 142, 150])
    x = np.arange(len(q))
    width = 0.25
    fig, ax = plt.subplots(figsize=(15, 8), facecolor=NAVY)
    title(fig, "Cloud Infrastructure ARR: AWS vs Azure vs Google Cloud", "Quarterly run-rate revenue in billions, with CAGR and total-change labels")
    style(ax)
    b1 = ax.bar(x - width, gcp, width, color=BLUE, label=f"Google Cloud ARR | total change {total_change(gcp):.1%} | CAGR {cagr(gcp):.1%}")
    b2 = ax.bar(x, azure, width, color="#f0522f", label=f"Azure ARR | total change {total_change(azure):.1%} | CAGR {cagr(azure):.1%}")
    b3 = ax.bar(x + width, aws, width, color=ORANGE, label=f"AWS ARR | total change {total_change(aws):.1%} | CAGR {cagr(aws):.1%}")
    annotate_bars(ax, [b1, b2, b3], fontsize=8.2)
    ax.set_xticks(x, q, rotation=0)
    ax.set_ylim(0, 178)
    ax.yaxis.tick_right()
    ax.set_ylabel("$bn ARR", rotation=270, labelpad=18)
    legend_below(ax, 1)
    save(fig, out)


def plot_quarterly_revenue_single(out: Path) -> None:
    labels = ["Apr '18", "Jul '18", "Oct '18", "Jan '19", "Apr '19", "Jul '19", "Oct '19", "Jan '20", "Apr '20", "Jul '20", "Oct '20", "Jan '21", "Apr '21", "Jul '21", "Oct '21", "Jan '22", "Apr '22", "Jul '22", "Oct '22", "Jan '23", "Apr '23", "Jul '23", "Oct '23", "Jan '24", "Apr '24", "Jul '24", "Oct '24", "Jan '25", "Apr '25", "Jul '25", "Oct '25", "Jan '26", "Apr '26"]
    revenue = np.array([47, 56, 66, 80, 96, 108, 125, 152, 178, 199, 232, 265, 303, 338, 380, 431, 488, 535, 581, 637, 693, 732, 786, 845, 921, 964, 1010, 1059, 1103, 1169, 1234, 1305, 1386])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(15, 8), facecolor=NAVY)
    title(fig, "Cybersecurity Revenue Run-Rate: Quarterly Revenue", "Statement-style revenue bars with absolute values, total change and CAGR")
    style(ax)
    bars = ax.bar(x, revenue, color=RED, width=0.72, label=f"Total revenue, quarterly ($m) | total change {total_change(revenue):.1%} | CAGR {cagr(revenue):.1%}")
    annotate_bars(ax, [bars], fontsize=7.8)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.yaxis.tick_right()
    ax.set_ylim(0, 1600)
    legend_below(ax)
    save(fig, out)


def plot_revenue_operating_income(out: Path) -> None:
    labels = quarter_labels(2020, 2, 25)
    revenue = np.array([54, 54, 66, 76, 88, 101, 114, 126, 143, 154, 170, 187, 204, 219, 238, 276, 281, 300, 322, 346, 367, 391, 416, 444, 479])
    operating = np.array([-60, -60, -54, -35, -39, -31, -32, -252, -70, -66, -63, -60, -76, -70, -55, -123, -66, -58, -47, -18, -33, -27, -2, 9, 7])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(15, 8), facecolor=NAVY)
    title(fig, "Revenue and Operating Income: Scaling Toward Profitability", "Positive revenue bars and operating-income bars around the zero line")
    style(ax)
    b_rev = ax.bar(x - 0.18, revenue, width=0.36, color="#07344c", label=f"Revenue ($m) | total change {total_change(revenue):.1%} | CAGR {cagr(revenue):.1%}")
    b_op = ax.bar(x + 0.18, operating, width=0.36, color="#2f81e7", label="Operating income ($m)")
    annotate_bars(ax, [b_rev, b_op], fontsize=7.4)
    ax.axhline(0, color=TEXT, linewidth=1.0, alpha=0.7)
    ax.set_xticks(x, labels, rotation=55, ha="right")
    ax.yaxis.tick_right()
    ax.set_ylim(-330, 620)
    legend_below(ax)
    save(fig, out)


def plot_ratio_zscore(out: Path, *, filename_title: str, subtitle: str, numerator_name: str, denominator_name: str, numerator: np.ndarray, denominator: np.ndarray, labels: list[str], color: str) -> None:
    ratio = pd.Series(numerator / denominator, index=labels)
    z = rolling_zscore(ratio, 8)
    x = np.arange(len(labels))
    fig = plt.figure(figsize=(14, 8), facecolor=NAVY)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 0.85])
    title(fig, filename_title, subtitle)
    ax_ratio = fig.add_subplot(gs[0])
    ax_z = fig.add_subplot(gs[1], sharex=ax_ratio)
    style(ax_ratio)
    style(ax_z)
    bars = ax_ratio.bar(x, ratio.values, width=0.70, color=color, alpha=0.86, label=f"{numerator_name} / {denominator_name}")
    ax_ratio.plot(x, ratio.rolling(4, min_periods=2).mean(), color=TEXT, linewidth=2.0, label="4Q average")
    ax_ratio.axhline(ratio.mean(), color=AMBER, linestyle=(0, (4, 4)), linewidth=1.1, label=f"sample average {ratio.mean():.2f}")
    if len(labels) <= 36:
        annotate_bars(ax_ratio, [bars], fontsize=8.0)
    else:
        annotate_sparse_bars(ax_ratio, bars, ratio.values, step=max(4, math.ceil(len(labels) / 16)))
    ax_ratio.set_ylabel("ratio")
    ax_ratio.legend(loc="upper left", frameon=False, labelcolor=TEXT, fontsize=8)

    z_colors = [GREEN if value > 0 else PINK for value in z.fillna(0)]
    ax_z.bar(x, z.fillna(0), color=z_colors, width=0.70, alpha=0.88)
    ax_z.axhline(0, color=TEXT, linewidth=1.0, alpha=0.55)
    ax_z.axhline(2, color=RED, linewidth=0.9, linestyle="--", alpha=0.75)
    ax_z.axhline(-2, color=CYAN, linewidth=0.9, linestyle="--", alpha=0.75)
    ax_z.set_ylabel("z-score")
    tick_idx, tick_labels = sampled_ticks(labels)
    ax_z.set_xticks(tick_idx, tick_labels, rotation=45, ha="right")
    latest = z.dropna().iloc[-1]
    ax_z.text(0.99, 0.88, f"latest z-score {latest:+.2f}", transform=ax_z.transAxes, ha="right", va="top", color=TEXT, fontsize=10, bbox={"facecolor": "#071a3d", "edgecolor": "#244b86", "alpha": 0.9, "pad": 4})
    save(fig, out)


def plot_margin_efficiency(out: Path) -> None:
    labels = quarter_labels(2021, 1, 22)
    revenue = np.linspace(88, 479, len(labels))
    gross_profit = revenue * np.linspace(0.66, 0.76, len(labels))
    operating_income = np.array([-39, -31, -32, -252, -70, -66, -63, -60, -76, -70, -55, -123, -66, -58, -47, -18, -33, -27, -2, 9, 7, 18])
    numerator = operating_income + gross_profit
    denominator = revenue
    plot_ratio_zscore(
        out,
        filename_title="Margin Efficiency: Operating Leverage Ratio + Z-Score",
        subtitle="(Gross profit + operating income) divided by revenue, then standardized over a rolling 8-quarter window",
        numerator_name="gross profit + operating income",
        denominator_name="revenue",
        numerator=numerator,
        denominator=denominator,
        labels=labels,
        color=CYAN,
    )


def plot_ps_ratio_zscore(out: Path) -> None:
    labels = quarter_labels(2021, 1, 22)
    revenue = np.linspace(1.2, 6.8, len(labels))
    market_cap = np.array([34, 38, 42, 48, 52, 45, 40, 47, 55, 61, 58, 72, 80, 86, 92, 88, 96, 110, 122, 135, 148, 166])
    plot_ratio_zscore(
        out,
        filename_title="Valuation Ratio: Market Cap / Forward Revenue + Z-Score",
        subtitle="A valuation multiple view built as numerator / denominator with rolling z-score regime bands",
        numerator_name="market cap",
        denominator_name="forward revenue",
        numerator=market_cap,
        denominator=revenue,
        labels=labels,
        color=PINK,
    )


def plot_short_volume_zscore(out: Path) -> None:
    labels = pd.date_range("2025-01-03", periods=64, freq="W-FRI").strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(31)
    total_volume = 85 + np.sin(np.linspace(0, 9, len(labels))) * 11 + rng.normal(0, 3.0, len(labels))
    short_volume = total_volume * (0.28 + np.sin(np.linspace(1, 11, len(labels))) * 0.055 + rng.normal(0, 0.015, len(labels)))
    plot_ratio_zscore(
        out,
        filename_title="FINRA Microstructure: Short Volume / Total Volume + Z-Score",
        subtitle="Short-volume pressure expressed as a ratio and standardized into crowded-trade state",
        numerator_name="short volume",
        denominator_name="total volume",
        numerator=short_volume,
        denominator=total_volume,
        labels=labels,
        color=PINK,
    )


def plot_credit_spread_ratio(out: Path) -> None:
    labels = pd.date_range("2023-01-31", periods=30, freq="ME").strftime("%b '%y").tolist()
    x = np.linspace(0, 9, len(labels))
    hy_oas = 3.8 + np.sin(x) * 0.85 + np.maximum(0, np.sin(x * 0.47 + 1)) * 0.45
    ten_y = 4.1 + np.cos(x * 0.6) * 0.42
    plot_ratio_zscore(
        out,
        filename_title="Credit Stress Ratio: High Yield OAS / 10Y Treasury + Z-Score",
        subtitle="Credit compensation relative to Treasury yield, with rolling z-score for spread-regime monitoring",
        numerator_name="HY OAS",
        denominator_name="10Y Treasury yield",
        numerator=hy_oas,
        denominator=ten_y,
        labels=labels,
        color=ORANGE,
    )


def plot_vix_vvix_ratio(out: Path) -> None:
    labels = pd.date_range("2024-01-05", periods=64, freq="W-FRI").strftime("%Y-%m-%d").tolist()
    x = np.linspace(0, 12, len(labels))
    vix = 17 + np.sin(x) * 5 + np.maximum(0, np.sin(x * 0.7 + 2)) * 7
    vvix = 82 + np.cos(x * 0.8) * 9 + np.maximum(0, np.sin(x * 0.55 + 1)) * 18
    plot_ratio_zscore(
        out,
        filename_title="Volatility-of-Vol Ratio: VVIX / VIX + Z-Score",
        subtitle="CBOE volatility-of-vol relative to spot VIX, standardized for convexity and hedge-cost review",
        numerator_name="VVIX",
        denominator_name="VIX",
        numerator=vvix,
        denominator=vix,
        labels=labels,
        color=CYAN,
    )


def plot_oil_curve_ratio(out: Path) -> None:
    labels = pd.date_range("2023-01-31", periods=34, freq="ME").strftime("%b '%y").tolist()
    x = np.linspace(0, 10, len(labels))
    cl1 = 76 + np.sin(x) * 12 + np.cos(x * 0.55) * 5
    cl2 = cl1 - 1.8 + np.sin(x * 0.8 + 2) * 3.6
    plot_ratio_zscore(
        out,
        filename_title="Oil Curve Ratio: CL1 / CL2 + Z-Score",
        subtitle="Front-month WTI relative to second contract, with z-score separating backwardation and contango pressure",
        numerator_name="CL1 front contract",
        denominator_name="CL2 second contract",
        numerator=cl1,
        denominator=cl2,
        labels=labels,
        color=GREEN,
    )


def plot_revision_breadth(out: Path) -> None:
    labels = quarter_labels(2022, 1, 18)
    positive = np.array([18, 20, 22, 25, 21, 28, 34, 38, 42, 39, 46, 51, 48, 54, 59, 63, 66, 72])
    negative = np.array([26, 24, 27, 31, 29, 24, 21, 19, 22, 25, 18, 16, 20, 17, 15, 13, 16, 12])
    breadth = (positive - negative) / (positive + negative)
    z = rolling_zscore(pd.Series(breadth), 6).fillna(0)
    x = np.arange(len(labels))
    fig = plt.figure(figsize=(14, 8), facecolor=NAVY)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 0.85])
    title(fig, "Earnings Revisions: Positive / Negative Revision Breadth", "Analyst estimate revisions converted into breadth, ratio and z-score momentum")
    ax = fig.add_subplot(gs[0])
    axz = fig.add_subplot(gs[1], sharex=ax)
    style(ax)
    style(axz)
    b_pos = ax.bar(x - 0.18, positive, width=0.36, color=GREEN, label="positive revisions")
    b_neg = ax.bar(x + 0.18, -negative, width=0.36, color=PINK, label="negative revisions")
    annotate_bars(ax, [b_pos, b_neg], fontsize=7.6)
    ax.axhline(0, color=TEXT, alpha=0.55)
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT)
    axz.bar(x, z, color=[GREEN if value > 0 else PINK for value in z], width=0.7)
    axz.axhline(0, color=TEXT, alpha=0.55)
    axz.axhline(2, color=RED, linestyle="--", alpha=0.65)
    axz.axhline(-2, color=CYAN, linestyle="--", alpha=0.65)
    axz.set_ylabel("breadth z")
    axz.set_xticks(x, labels, rotation=45, ha="right")
    save(fig, out)


def plot_roic_wacc_spread(out: Path) -> None:
    labels = quarter_labels(2021, 1, 22)
    roic = 9.0 + np.linspace(0, 8, len(labels)) + np.sin(np.linspace(0, 8, len(labels))) * 1.4
    wacc = 8.5 + np.cos(np.linspace(0, 5, len(labels))) * 0.8
    spread = roic - wacc
    z = rolling_zscore(pd.Series(spread), 8).fillna(0)
    x = np.arange(len(labels))
    fig = plt.figure(figsize=(14, 8), facecolor=NAVY)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 0.85])
    title(fig, "Capital Quality: ROIC - WACC Spread + Z-Score", "Return on invested capital relative to cost of capital, with spread regime standardized")
    ax = fig.add_subplot(gs[0])
    axz = fig.add_subplot(gs[1], sharex=ax)
    style(ax)
    style(axz)
    b1 = ax.bar(x - 0.18, roic, width=0.36, color=GREEN, label="ROIC")
    b2 = ax.bar(x + 0.18, wacc, width=0.36, color=AMBER, label="WACC")
    ax.plot(x, spread, color=TEXT, linewidth=2.2, marker="o", markersize=3.5, label="ROIC - WACC spread")
    annotate_bars(ax, [b1, b2], fontsize=7.4, as_pct=True)
    ax.set_ylabel("%")
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT, fontsize=8)
    axz.bar(x, z, color=[GREEN if value > 0 else PINK for value in z], width=0.7)
    axz.axhline(0, color=TEXT, alpha=0.55)
    axz.axhline(2, color=RED, linestyle="--", alpha=0.65)
    axz.axhline(-2, color=CYAN, linestyle="--", alpha=0.65)
    axz.set_ylabel("spread z")
    axz.set_xticks(x, labels, rotation=45, ha="right")
    save(fig, out)


def plot_margin_waterfall(out: Path) -> None:
    labels = ["Revenue growth", "Gross margin", "S&M leverage", "R&D leverage", "G&A leverage", "Stock comp", "FCF margin"]
    values = np.array([18.2, 4.4, 6.1, 2.2, 1.8, -3.7, 28.9])
    starts = np.r_[0, np.cumsum(values[:-1])]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=NAVY)
    title(fig, "Unit Economics Waterfall: Revenue Growth to FCF Margin", "Statement lines converted into a bridge showing operating leverage and cost drag")
    style(ax)
    colors = [GREEN if v >= 0 else PINK for v in values]
    bars = ax.bar(x, values, bottom=starts, color=colors, width=0.65)
    ax.plot(x, starts + values, color=TEXT, alpha=0.45, linewidth=1.1)
    ax.axhline(0, color=TEXT, alpha=0.55)
    for i, (base, value) in enumerate(zip(starts, values)):
        ax.text(i, base + value / 2, f"{value:+.1f} pts", color=TEXT, ha="center", va="center", fontsize=9, weight="semibold")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("margin points")
    _ = bars
    save(fig, out)


def plot_investment_evidence_timeline(out: Path) -> None:
    rng = np.random.default_rng(1301)
    idx = pd.date_range("2025-01-02", periods=260, freq="B")
    price = pd.Series(np.cumprod(1 + rng.normal(0.00055, 0.014, len(idx))) * 188, index=idx)
    sma63 = price.rolling(63, min_periods=20).mean()
    events = [
        ("SEC 10-K", idx[34], "SEC filings", BLUE),
        ("Form 4", idx[58], "Insiders", PINK),
        ("Earnings", idx[83], "Earnings", GREEN),
        ("Revision +", idx[118], "Estimates", CYAN),
        ("FINRA spike", idx[151], "Short volume", AMBER),
        ("SEC 10-Q", idx[188], "SEC filings", BLUE),
        ("VIX high", idx[220], "CBOE VIX", RED),
    ]
    lanes = ["SEC filings", "Insiders", "Earnings", "Estimates", "Short volume", "CBOE VIX"]
    fig = plt.figure(figsize=(15, 8), facecolor=NAVY)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 0.75])
    title(fig, "Investment Evidence Timeline: AAPL Multi-Source Packet", "Pricing, SEC filings, Form 4 insiders, earnings, revisions, FINRA shorts and VIX on one audit-ready timeline")
    ax_price = fig.add_subplot(gs[0])
    ax_lane = fig.add_subplot(gs[1], sharex=ax_price)
    style(ax_price)
    style(ax_lane)
    ax_price.plot(idx, price, color=CYAN, linewidth=2.0, label="adjusted close")
    ax_price.plot(idx, sma63, color=TEXT, linewidth=1.4, linestyle="--", label="SMA 63")
    ax_price.fill_between(idx, sma63, price, where=price >= sma63, color=GREEN, alpha=0.10)
    ax_price.fill_between(idx, sma63, price, where=price < sma63, color=PINK, alpha=0.12)
    for label, event_date, lane, color in events:
        ax_price.axvline(event_date, color=color, alpha=0.55, linewidth=1.0)
        ax_price.text(event_date, price.max() * 1.01, label, rotation=90, color=TEXT, fontsize=7, va="bottom", ha="center")
    ax_price.set_ylabel("price")
    ax_price.legend(loc="upper left", frameon=False, labelcolor=TEXT, fontsize=8)

    ax_lane.set_ylim(-0.5, len(lanes) - 0.5)
    ax_lane.set_yticks(range(len(lanes)), lanes)
    ax_lane.grid(True, axis="x", color=GRID, alpha=0.35)
    ax_lane.grid(False, axis="y")
    for label, event_date, lane, color in events:
        y = lanes.index(lane)
        ax_lane.scatter(event_date, y, s=230, color=color, edgecolors=TEXT, linewidths=0.6, zorder=5)
        ax_lane.text(event_date, y + 0.23, label, color=TEXT, fontsize=7.2, ha="center", va="bottom")
    metrics = [("sources", "7"), ("events", str(len(events))), ("last", fmt(float(price.iloc[-1]))), ("63D trend", "above" if price.iloc[-1] > sma63.iloc[-1] else "below")]
    for i, (label, value) in enumerate(metrics):
        ax_price.text(0.68 + i * 0.075, 0.05, label.upper() + "\n" + value, transform=ax_price.transAxes, color=TEXT, fontsize=8, ha="center", bbox={"facecolor": "#071a3d", "edgecolor": "#244b86", "alpha": 0.88, "pad": 4})
    save(fig, out)


def plot_earnings_implied_realized(out: Path) -> None:
    symbols = ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "CRM", "ADBE", "SNOW", "CRWD", "PANW", "AVGO"]
    surprise = np.array([2.1, 1.4, 5.8, 3.1, 0.8, 1.9, -0.7, 0.4, 4.6, 6.2, 2.8, 3.9])
    realized = np.array([1.6, 2.2, 8.9, 4.2, -0.5, 3.4, -4.1, -1.8, 6.7, 9.4, 3.1, 5.8])
    implied = np.array([3.2, 3.6, 7.1, 5.0, 3.0, 4.3, 5.7, 4.2, 8.2, 9.5, 6.4, 5.8])
    vix = np.array([18, 19, 22, 20, 17, 18, 26, 24, 29, 31, 23, 21])
    fig, ax = plt.subplots(figsize=(13.5, 8), facecolor=NAVY)
    title(fig, "Earnings Surprise: Implied Move vs Realized Move", "EPS surprise, option-implied move, realized 5D return and VIX regime combined in one event-study view")
    style(ax)
    colors = [RED if x >= 26 else AMBER if x >= 22 else CYAN for x in vix]
    sizes = 85 + implied * 28
    ax.scatter(surprise, realized, s=sizes, color=colors, edgecolors=TEXT, linewidths=0.55, alpha=0.88)
    ax.axhline(0, color=TEXT, alpha=0.45)
    ax.axvline(0, color=TEXT, alpha=0.45)
    xline = np.linspace(surprise.min() - 1, surprise.max() + 1, 80)
    ax.plot(xline, xline, color=GREEN, linestyle="--", linewidth=1.1, label="realized = EPS surprise")
    for sym, x, y, imp in zip(symbols, surprise, realized, implied):
        ax.text(x + 0.08, y + 0.14, f"{sym}\n{imp:.1f}% imp", color=TEXT, fontsize=7.2)
    ax.set_xlabel("EPS surprise, %")
    ax.set_ylabel("realized 5D return, %")
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT)
    ax.text(0.99, 0.05, "bubble size = option implied move | color = VIX regime", transform=ax.transAxes, ha="right", color=MUTED, fontsize=8)
    save(fig, out)


def plot_crowding_capacity_map(out: Path) -> None:
    names = ["NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "LLY", "JPM", "XOM", "AVGO", "COST", "TSLA"]
    days = np.array([7.2, 4.1, 4.6, 5.4, 5.0, 6.2, 12.8, 3.4, 2.9, 9.6, 8.8, 15.2])
    concentration = np.array([0.86, 0.62, 0.58, 0.67, 0.55, 0.61, 0.78, 0.44, 0.39, 0.81, 0.72, 0.90])
    position_adv = np.array([1.8, 0.9, 1.1, 1.2, 0.8, 1.4, 2.1, 0.6, 0.5, 1.9, 1.6, 2.4])
    short_z = np.array([1.5, 0.2, 0.4, 0.8, -0.1, 0.6, 1.2, -0.4, -0.6, 1.0, 0.3, 2.2])
    fig, ax = plt.subplots(figsize=(13.5, 8), facecolor=NAVY)
    title(fig, "Crowding + Liquidity Capacity Map", "13F concentration, FINRA short pressure, ADV capacity and price volatility converted into one liquidity screen")
    style(ax)
    scatter = ax.scatter(days, concentration, s=position_adv * 360, c=short_z, cmap=CMAP, vmin=-2.5, vmax=2.5, edgecolors=TEXT, linewidths=0.55, alpha=0.90)
    ax.axvline(10, color=AMBER, linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axhline(0.75, color=AMBER, linestyle="--", linewidth=1.0, alpha=0.75)
    for name, x, y in zip(names, days, concentration):
        ax.text(x + 0.22, y, name, color=TEXT, fontsize=8, va="center")
    ax.set_xlabel("estimated days to exit @ 5% ADV")
    ax.set_ylabel("institutional holder concentration")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.018)
    cbar.set_label("short-volume z-score", color=MUTED)
    cbar.ax.tick_params(colors=MUTED)
    ax.text(0.02, 0.05, "bubble size = position / ADV capacity", transform=ax.transAxes, color=MUTED, fontsize=8)
    save(fig, out)


def plot_macro_regime_cross_asset_matrix(out: Path) -> None:
    regimes = ["growth up\ninflation down", "growth up\ninflation up", "growth down\ninflation down", "growth down\ninflation up"]
    assets = ["SPY", "TLT", "GLD", "UUP", "BTC", "CL1", "HYG"]
    matrix = np.array([[0.84, 0.32, 0.22, -0.12, 0.91, 0.18, 0.57], [0.28, -0.46, 0.41, 0.36, 0.22, 0.76, 0.10], [0.12, 0.88, 0.52, -0.18, -0.35, -0.22, 0.33], [-0.62, 0.18, 0.71, 0.58, -0.82, 0.44, -0.55]])
    cot = pd.Series({"ES": 82, "CL": 68, "GC": 74, "ZN": 12})
    fig = plt.figure(figsize=(14, 8), facecolor=NAVY)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 0.55])
    title(fig, "Macro Regime Cross-Asset Matrix", "FRED macro regimes, CFTC COT percentiles and cross-asset returns compressed into one allocation surface")
    ax = fig.add_subplot(gs[0])
    ax_cot = fig.add_subplot(gs[1])
    style(ax)
    style(ax_cot)
    im = ax.imshow(matrix, cmap=CMAP, vmin=-1, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(assets)), assets)
    ax.set_yticks(range(len(regimes)), regimes)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", color=TEXT, ha="center", va="center", fontsize=8.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("forward Sharpe proxy", color=MUTED)
    cbar.ax.tick_params(colors=MUTED)
    bars = ax_cot.barh(cot.index, cot.values, color=[PINK if v > 80 or v < 20 else CYAN for v in cot.values], alpha=0.9)
    ax_cot.set_xlim(0, 100)
    ax_cot.set_title("CFTC COT percentile", color=TEXT, loc="left", fontsize=12)
    annotate_bars(ax_cot, [bars], fontsize=8)
    save(fig, out)


def plot_roic_wacc_quality_frontier(out: Path) -> None:
    names = ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "CRM", "ADBE", "AVGO", "LLY", "JPM", "XOM"]
    spread = np.array([18, 21, 25, 17, 16, 8, 7, 15, 19, 11, 6, 4])
    growth = np.array([5, 11, 68, 19, 14, 12, 9, 8, 32, 15, 4, 2])
    fcf_margin = np.array([24, 31, 39, 34, 26, 8, 21, 29, 36, 22, 18, 12])
    valuation_z = np.array([0.6, 0.8, 2.1, 0.4, 0.3, -0.1, -0.6, 0.2, 1.3, 1.6, -0.8, -1.0])
    fig, ax = plt.subplots(figsize=(13.5, 8), facecolor=NAVY)
    title(fig, "ROIC vs WACC Quality Frontier", "Fundamentals, FRED rates, valuation z-score and price momentum combined into a quality frontier")
    style(ax)
    scatter = ax.scatter(spread, growth, s=fcf_margin * 18, c=valuation_z, cmap=CMAP, vmin=-2.5, vmax=2.5, edgecolors=TEXT, linewidths=0.55, alpha=0.9)
    ax.axvline(10, color=AMBER, linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axhline(10, color=AMBER, linestyle="--", linewidth=1.0, alpha=0.75)
    for name, x, y in zip(names, spread, growth):
        ax.text(x + 0.35, y, name, color=TEXT, fontsize=8, va="center")
    ax.set_xlabel("ROIC - WACC spread, pts")
    ax.set_ylabel("revenue growth, %")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.018)
    cbar.set_label("valuation z-score", color=MUTED)
    cbar.ax.tick_params(colors=MUTED)
    ax.text(0.02, 0.05, "bubble size = FCF margin", transform=ax.transAxes, color=MUTED, fontsize=8)
    save(fig, out)


def plot_revisions_momentum_confluence(out: Path) -> None:
    names = ["NVDA", "AVGO", "MSFT", "META", "AAPL", "GOOGL", "AMZN", "CRM", "ADBE", "PANW"]
    revisions = np.array([2.0, 1.6, 0.8, 0.9, -0.2, 0.4, 0.7, -0.5, -0.8, 0.3])
    momentum = np.array([1.8, 1.4, 0.7, 1.0, -0.1, 0.2, 0.9, -0.4, -0.6, 0.5])
    surprise = np.array([1.5, 0.9, 0.4, 0.8, 0.1, 0.3, 0.5, -0.7, -0.4, 0.2])
    short_pressure = np.array([0.6, 0.4, 0.1, 0.2, 0.3, -0.2, 0.0, 0.8, 0.7, 0.4])
    data = pd.DataFrame({"revision breadth": revisions, "price momentum": momentum, "earnings surprise": surprise, "short pressure": short_pressure}, index=names)
    score = data[["revision breadth", "price momentum", "earnings surprise"]].mean(axis=1) - data["short pressure"] * 0.25
    fig = plt.figure(figsize=(14, 8), facecolor=NAVY)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8])
    title(fig, "Revisions + Momentum Confluence", "Analyst revisions, earnings surprise, price momentum and short pressure merged into a cross-sectional signal view")
    ax_h = fig.add_subplot(gs[0])
    ax_s = fig.add_subplot(gs[1])
    style(ax_h)
    style(ax_s)
    im = ax_h.imshow(data.values, cmap=CMAP, vmin=-2.2, vmax=2.2, aspect="auto")
    ax_h.grid(False)
    ax_h.set_xticks(range(data.shape[1]), data.columns, rotation=25, ha="right")
    ax_h.set_yticks(range(data.shape[0]), data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax_h.text(j, i, f"{data.iloc[i, j]:+.1f}", color=TEXT, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax_h, fraction=0.035, pad=0.02)
    ordered = score.sort_values()
    bars = ax_s.barh(ordered.index, ordered.values, color=[GREEN if v > 0 else PINK for v in ordered], alpha=0.92)
    ax_s.axvline(0, color=TEXT, alpha=0.55)
    ax_s.set_title("confluence score", color=TEXT, loc="left", fontsize=12)
    annotate_bars(ax_s, [bars], fontsize=8)
    save(fig, out)


def plot_security_master_replay(out: Path) -> None:
    rng = np.random.default_rng(77)
    idx = pd.date_range("2023-01-03", periods=360, freq="B")
    raw = pd.Series(np.cumprod(1 + rng.normal(0.00045, 0.013, len(idx))) * 112, index=idx)
    adjusted = raw.copy()
    split_date = idx[145]
    dividend_date = idx[235]
    ticker_date = idx[294]
    adjusted.loc[adjusted.index < split_date] *= 0.5
    adjusted.loc[adjusted.index < dividend_date] *= 0.985
    fig = plt.figure(figsize=(15, 8), facecolor=NAVY)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 0.55])
    title(fig, "Security Master + Corporate Actions Replay", "OpenFIGI identity, ticker mapping, split/dividend events and adjusted OHLCV shown as a replayable data contract")
    ax = fig.add_subplot(gs[0])
    ax_card = fig.add_subplot(gs[1])
    style(ax)
    style(ax_card)
    ax.plot(idx, raw, color=PINK, linewidth=1.8, label="raw close")
    ax.plot(idx, adjusted, color=CYAN, linewidth=2.0, label="split/dividend adjusted close")
    for date, label, color in [(split_date, "2-for-1 split", GREEN), (dividend_date, "dividend", AMBER), (ticker_date, "ticker mapping", BLUE)]:
        ax.axvline(date, color=color, alpha=0.72, linewidth=1.1)
        ax.text(date, raw.max() * 1.01, label, rotation=90, color=TEXT, fontsize=8, va="bottom", ha="center")
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT, fontsize=8)
    ax.set_ylabel("price")
    ax_card.axis("off")
    fields = [("ticker", "AAPL"), ("composite FIGI", "BBG000B9XRY4"), ("share class", "BBG001S5N8V8"), ("security type", "Common Stock"), ("currency", "USD"), ("adjustment policy", "split + dividend")]
    y = 0.84
    for label, value in fields:
        ax_card.text(0.05, y, label.upper(), transform=ax_card.transAxes, color=MUTED, fontsize=8)
        ax_card.text(0.05, y - 0.065, value, transform=ax_card.transAxes, color=TEXT, fontsize=14, weight="semibold")
        ax_card.plot([0.05, 0.92], [y - 0.105, y - 0.105], transform=ax_card.transAxes, color=GRID, alpha=0.7)
        y -= 0.14
    save(fig, out)


def plot_portfolio_shock_waterfall(out: Path) -> None:
    labels = ["Market beta", "Rates +35bp", "VIX +25pt", "USD +2%", "Oil +10%", "Idiosyncratic", "Total stress P&L"]
    shocks = np.array([-3.1, -1.2, -1.7, -0.5, 0.8, -0.9])
    total_pnl = shocks.sum()
    starts = np.r_[0, np.cumsum(shocks[:-1]), 0]
    heights = np.r_[shocks, total_pnl]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=NAVY)
    title(fig, "Portfolio Shock Waterfall", "Holdings, factor exposure, macro rates, VIX, USD and oil shocks converted into estimated book-level P&L")
    style(ax)
    colors = [GREEN if v > 0 else PINK for v in shocks] + [CYAN]
    ax.bar(x, heights, bottom=starts, color=colors, width=0.68, alpha=0.92)
    ax.plot(x, starts + heights, color=TEXT, alpha=0.5, linewidth=1.1)
    for i, (base, h) in enumerate(zip(starts, heights)):
        label = f"{h:+.1f}m" if i < len(labels) - 1 else f"{h:.1f}m"
        ax.text(i, base + h / 2, label, color=TEXT, ha="center", va="center", fontsize=9, weight="semibold")
    ax.axhline(0, color=TEXT, alpha=0.55)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("$m P&L")
    ax.set_ylim(min(starts + heights) - 1.0, 1.6)
    ax.text(0.70, 0.88, "BASE NAV\n$125.4m", transform=ax.transAxes, color=TEXT, fontsize=10, ha="center", bbox={"facecolor": "#071a3d", "edgecolor": "#244b86", "alpha": 0.9, "pad": 5})
    ax.text(0.84, 0.88, "STRESS NAV\n$118.8m", transform=ax.transAxes, color=TEXT, fontsize=10, ha="center", bbox={"facecolor": "#071a3d", "edgecolor": "#244b86", "alpha": 0.9, "pad": 5})
    ax.text(0.965, 0.88, f"NAV HIT\n{total_pnl / 125.4:.1%}", transform=ax.transAxes, color=TEXT, fontsize=10, ha="right", bbox={"facecolor": "#071a3d", "edgecolor": "#244b86", "alpha": 0.9, "pad": 5})
    save(fig, out)


def plot_provider_evidence_receipt(out: Path) -> None:
    rows = [
        ("pricing", "Tiingo / EOD", "equity.pricing", "252 rows", "ok"),
        ("fundamentals", "FMP", "equity.fundamentals", "38 metrics", "ok"),
        ("filings", "SEC EDGAR", "regulatory.sec", "10-K / 10-Q / Form 4", "ok"),
        ("identity", "OpenFIGI", "reference.identifiers", "FIGI + share class", "ok"),
        ("shorts", "FINRA", "market-structure.finra", "short-volume ratio", "ok"),
        ("volatility", "CBOE", "derivatives.vol", "VIX / VVIX / SKEW", "ok"),
        ("macro", "FRED", "macro.series", "DGS10 / CPI / UNRATE", "ok"),
    ]
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=NAVY)
    title(fig, "Provider Evidence Receipt", "A single research packet showing the providers, route families, data objects and execution status behind the final output")
    style(ax)
    ax.axis("off")
    headers = ["data object", "provider", "route family", "returned", "status"]
    widths = [0.19, 0.20, 0.25, 0.22, 0.10]
    x0 = 0.04
    y = 0.82
    for i, header in enumerate(headers):
        ax.text(x0 + sum(widths[:i]), y, header.upper(), transform=ax.transAxes, color=MUTED, fontsize=8, weight="semibold")
    y -= 0.06
    for row_index, row in enumerate(rows):
        bg = "#071a3d" if row_index % 2 == 0 else "#061226"
        ax.add_patch(plt.Rectangle((0.035, y - 0.032), 0.93, 0.055, transform=ax.transAxes, facecolor=bg, edgecolor="#17376f", linewidth=0.5, alpha=0.92))
        for i, value in enumerate(row):
            color = GREEN if i == 4 else TEXT
            ax.text(x0 + sum(widths[:i]), y, value, transform=ax.transAxes, color=color, fontsize=10, va="center")
        y -= 0.075
    metrics = [("providers", "7"), ("route families", "7"), ("objects", "13"), ("warnings", "0")]
    for i, (label, value) in enumerate(metrics):
        ax.text(0.09 + i * 0.22, 0.12, label.upper(), transform=ax.transAxes, color=MUTED, fontsize=8)
        ax.text(0.09 + i * 0.22, 0.065, value, transform=ax.transAxes, color=TEXT, fontsize=24, weight="semibold")
    save(fig, out)


def plot_factor_exposure_drift(out: Path) -> None:
    idx = pd.date_range("2024-01-31", periods=30, freq="ME")
    x = np.linspace(0, 7, len(idx))
    exposures = pd.DataFrame(
        {
            "Market": 0.55 + np.sin(x) * 0.08,
            "Momentum": 0.18 + np.cos(x * 0.8) * 0.05,
            "Quality": 0.16 + np.sin(x * 0.6 + 1) * 0.04,
            "Rates": -0.10 + np.cos(x * 0.9) * 0.06,
            "Oil": 0.05 + np.sin(x * 1.1) * 0.04,
            "USD": -0.06 + np.cos(x * 0.7 + 1) * 0.03,
        },
        index=idx,
    )
    beta_z = rolling_zscore(exposures["Market"], 8).fillna(0)
    fig = plt.figure(figsize=(14, 8), facecolor=NAVY)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 0.8])
    title(fig, "Factor Exposure Drift", "Portfolio holdings, adjusted returns, Fama-French factors, macro rates and VIX state shown as exposure drift over time")
    ax = fig.add_subplot(gs[0])
    axz = fig.add_subplot(gs[1], sharex=ax)
    style(ax)
    style(axz)
    colors = [BLUE, CYAN, GREEN, PINK, AMBER, RED]
    ax.stackplot(idx, [exposures[col] for col in exposures.columns], labels=exposures.columns, colors=colors, alpha=0.78)
    ax.axhline(0, color=TEXT, alpha=0.45)
    ax.set_ylabel("exposure")
    ax.legend(loc="upper left", ncol=3, frameon=False, labelcolor=TEXT, fontsize=8)
    axz.bar(idx, beta_z, width=20, color=[GREEN if v > 0 else PINK for v in beta_z], alpha=0.9)
    axz.axhline(0, color=TEXT, alpha=0.55)
    axz.axhline(2, color=RED, linestyle="--", alpha=0.65)
    axz.axhline(-2, color=CYAN, linestyle="--", alpha=0.65)
    axz.set_ylabel("market beta z")
    save(fig, out)


def plot_orange_factor_risk_river(out: Path) -> None:
    idx = pd.date_range("2023-01-31", periods=38, freq="ME")
    x = np.linspace(0, 9, len(idx))
    exposures = pd.DataFrame(
        {
            "Momentum": 0.18 + np.maximum(0, np.sin(x * 0.8)) * 0.22,
            "Quality": 0.20 + np.maximum(0, np.cos(x * 0.55 + 1)) * 0.18,
            "Carry": 0.08 + np.maximum(0, np.sin(x * 0.65 + 2)) * 0.14,
            "Rates hedge": 0.10 + np.maximum(0, np.cos(x * 0.9)) * 0.16,
            "Vol hedge": 0.06 + np.maximum(0, np.sin(x * 1.1 + 1.3)) * 0.15,
        },
        index=idx,
    )
    exposures = exposures.div(exposures.sum(axis=1), axis=0)
    signal = rolling_zscore(exposures["Momentum"] - exposures["Rates hedge"], 8).fillna(0)
    fig = plt.figure(figsize=(14, 8), facecolor=NAVY)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 0.75])
    title(fig, "Factor Risk River: Allocation Drift by Signal Sleeve", "Fama-French style factors, macro rates and volatility hedge inputs visualized as a time-varying exposure river")
    ax = fig.add_subplot(gs[0])
    axz = fig.add_subplot(gs[1], sharex=ax)
    style(ax)
    style(axz)
    colors = [ORANGE, AMBER, GREEN, CYAN, PINK]
    ax.stackplot(idx, [exposures[col] for col in exposures.columns], labels=exposures.columns, colors=colors, alpha=0.86)
    ax.set_ylabel("portfolio sleeve weight")
    ax.legend(loc="upper left", ncol=3, frameon=False, labelcolor=TEXT, fontsize=8)
    axz.bar(idx, signal, width=20, color=[ORANGE if v > 0 else PINK for v in signal], alpha=0.92)
    axz.axhline(0, color=TEXT, alpha=0.55)
    axz.axhline(2, color=RED, linestyle="--", alpha=0.65)
    axz.axhline(-2, color=CYAN, linestyle="--", alpha=0.65)
    axz.set_ylabel("momentum - rates z")
    save(fig, out)


def plot_options_vol_surface_3d(out: Path) -> None:
    expiries = np.array([7, 14, 30, 60, 90, 180, 365])
    moneyness = np.linspace(0.75, 1.25, 21)
    X, Y = np.meshgrid(moneyness, expiries)
    surface = 0.18 + 0.22 * (X - 1.0) ** 2 + 0.07 * np.exp(-Y / 45) + 0.035 * np.maximum(0, 1 - X)
    fig = plt.figure(figsize=(14, 8), facecolor=NAVY)
    title(fig, "SPY Options Surface: Implied Volatility by Strike and Expiry", "Representative surface from SPY option chain, expiries and vol context.")
    ax, cax = surface_layout(fig)
    style_3d(ax)
    surf = ax.plot_surface(X, Y, surface * 100, cmap=LinearSegmentedColormap.from_list("vol3d", [CYAN, ORANGE, PINK]), linewidth=0.15, edgecolor="#17376f", antialiased=True, alpha=0.95)
    ax.contour(X, Y, surface * 100, zdir="z", offset=surface.min() * 100 - 1, cmap=CMAP, linewidths=1.0)
    ax.set_xlabel("moneyness")
    ax.set_ylabel("days to expiry")
    ax.set_zlabel("implied vol, %")
    ax.view_init(elev=27, azim=-130)
    cbar = fig.colorbar(surf, cax=cax)
    cbar.set_label("IV %", color=MUTED)
    cbar.ax.tick_params(colors=MUTED)
    save(fig, out)


def plot_backtest_parameter_surface_3d(out: Path) -> None:
    fast = np.array([5, 10, 15, 20, 30, 40, 60])
    slow = np.array([80, 100, 125, 160, 200, 250, 320])
    X, Y = np.meshgrid(fast, slow)
    surface = 0.65 + 0.55 * np.exp(-((X - 20) ** 2 / 420 + (Y - 180) ** 2 / 9000)) - 0.16 * np.abs(X / Y - 0.12)
    surface += 0.05 * np.sin(X / 8) + 0.04 * np.cos(Y / 45)
    best = np.unravel_index(np.argmax(surface), surface.shape)
    fig = plt.figure(figsize=(14, 8), facecolor=NAVY)
    title(fig, "SPY Backtest Surface: SMA Fast / Slow Parameter Robustness", "Representative SPY SMA-grid backtest surface from adjusted OHLCV.")
    ax, cax = surface_layout(fig)
    style_3d(ax)
    surf = ax.plot_surface(X, Y, surface, cmap=LinearSegmentedColormap.from_list("bt3d", [PINK, ORANGE, GREEN]), edgecolor="#17376f", linewidth=0.2, alpha=0.96)
    ax.scatter([X[best]], [Y[best]], [surface[best]], color=TEXT, s=80, edgecolors=INK, linewidths=0.6)
    ax.text(X[best], Y[best], surface[best] + 0.05, f"best {surface[best]:.2f}", color=TEXT, fontsize=8)
    ax.set_xlabel("fast SMA")
    ax.set_ylabel("slow SMA")
    ax.set_zlabel("Sharpe")
    ax.view_init(elev=28, azim=-128)
    cbar = fig.colorbar(surf, cax=cax)
    cbar.set_label("Sharpe", color=MUTED)
    cbar.ax.tick_params(colors=MUTED)
    save(fig, out)


def plot_technical_indicator_stack(out: Path) -> None:
    rng = np.random.default_rng(2401)
    idx = pd.date_range("2025-01-02", periods=260, freq="B")
    price = pd.Series(np.cumprod(1 + rng.normal(0.0006, 0.013, len(idx))) * 210, index=idx)
    sma20 = price.rolling(20).mean()
    sma50 = price.rolling(50).mean()
    std20 = price.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    ret = price.pct_change()
    rsi = 50 + 32 * np.tanh((ret.rolling(14).mean() / ret.rolling(14).std()).fillna(0))
    ema12 = price.ewm(span=12).mean()
    ema26 = price.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    vol = ret.rolling(21).std() * np.sqrt(252) * 100
    fig = plt.figure(figsize=(14, 9), facecolor=NAVY)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.55, 0.75, 0.75, 0.75])
    title(fig, "Technical Indicator Stack: Trend, Momentum, Volatility", "Adjusted OHLCV converted into Bollinger bands, SMA trend, RSI, MACD and realized-volatility state")
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in axes:
        style(ax)
    axes[0].plot(idx, price, color=CYAN, linewidth=1.8, label="adjusted close")
    axes[0].plot(idx, sma20, color=ORANGE, linewidth=1.3, label="SMA 20")
    axes[0].plot(idx, sma50, color=TEXT, linewidth=1.2, linestyle="--", label="SMA 50")
    axes[0].fill_between(idx, lower, upper, color=BLUE, alpha=0.10, label="Bollinger 20D 2σ")
    axes[0].legend(loc="upper left", ncol=4, frameon=False, labelcolor=TEXT, fontsize=8)
    axes[1].plot(idx, rsi, color=AMBER, linewidth=1.7)
    axes[1].axhline(70, color=RED, linestyle="--", alpha=0.7)
    axes[1].axhline(30, color=CYAN, linestyle="--", alpha=0.7)
    axes[1].set_ylabel("RSI")
    axes[2].bar(idx, macd - signal, width=1.0, color=[GREEN if v > 0 else PINK for v in (macd - signal).fillna(0)], alpha=0.8)
    axes[2].plot(idx, macd, color=ORANGE, linewidth=1.2)
    axes[2].plot(idx, signal, color=TEXT, linewidth=1.1)
    axes[2].set_ylabel("MACD")
    axes[3].plot(idx, vol, color=PINK, linewidth=1.6)
    axes[3].fill_between(idx, vol, color=PINK, alpha=0.13)
    axes[3].set_ylabel("21D vol %")
    latest = price.iloc[-1]
    axes[0].text(0.985, 0.08, f"last {latest:.2f} | SMA20 {sma20.iloc[-1]:.2f} | RSI {rsi.iloc[-1]:.1f}", transform=axes[0].transAxes, ha="right", color=TEXT, fontsize=8, bbox={"facecolor": "#071a3d", "edgecolor": "#244b86", "alpha": 0.9, "pad": 4})
    save(fig, out)


def plot_volatility_cone(out: Path) -> None:
    horizons = np.array([5, 10, 21, 42, 63, 126, 252])
    p10 = np.array([12, 13, 14, 15, 16, 17, 18])
    p50 = np.array([20, 21, 22, 23, 24, 25, 26])
    p90 = np.array([36, 34, 33, 32, 31, 30, 29])
    current = np.array([41, 38, 34, 29, 27, 24, 23])
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=NAVY)
    title(fig, "Realized Volatility Cone", "Historical volatility percentiles and current realized-vol state across multiple lookback windows")
    style(ax)
    ax.fill_between(horizons, p10, p90, color=BLUE, alpha=0.16, label="10-90 percentile cone")
    ax.fill_between(horizons, p10, p50, color=CYAN, alpha=0.10)
    ax.plot(horizons, p50, color=TEXT, linewidth=2.0, label="median")
    ax.plot(horizons, current, color=ORANGE, marker="o", linewidth=2.4, label="current realized vol")
    for x, y in zip(horizons, current):
        ax.text(x, y + 1.1, f"{y:.0f}%", color=TEXT, fontsize=8, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(horizons, [f"{h}D" for h in horizons])
    ax.set_ylabel("annualized realized volatility")
    ax.legend(loc="upper right", frameon=False, labelcolor=TEXT)
    save(fig, out)


def plot_regime_allocation_river(out: Path) -> None:
    idx = pd.date_range("2023-01-31", periods=40, freq="ME")
    x = np.linspace(0, 10, len(idx))
    weights = pd.DataFrame(
        {
            "Equity": 0.35 + 0.10 * np.sin(x),
            "Duration": 0.22 + 0.10 * np.cos(x * 0.7 + 1),
            "Gold": 0.13 + 0.05 * np.sin(x * 1.3 + 2),
            "Commodities": 0.12 + 0.07 * np.maximum(0, np.sin(x * 0.9 + 1)),
            "USD cash": 0.10 + 0.04 * np.cos(x * 1.1),
        },
        index=idx,
    )
    weights = weights.clip(0.03).div(weights.clip(0.03).sum(axis=1), axis=0)
    cot_state = 50 + 35 * np.sin(x * 0.75 + 1)
    fig = plt.figure(figsize=(14, 8), facecolor=NAVY)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 0.65])
    title(fig, "Macro Regime Allocation River", "FRED growth/inflation state, CFTC positioning and cross-asset pricing translated into allocation weights")
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    style(ax)
    style(ax2)
    ax.stackplot(idx, [weights[col] for col in weights.columns], labels=weights.columns, colors=[ORANGE, CYAN, AMBER, GREEN, PINK], alpha=0.84)
    ax.set_ylabel("target weight")
    ax.legend(loc="upper left", ncol=5, frameon=False, labelcolor=TEXT, fontsize=8)
    ax2.plot(idx, cot_state, color=ORANGE, linewidth=2.0, label="COT risk-on percentile")
    ax2.axhline(80, color=RED, linestyle="--", alpha=0.65)
    ax2.axhline(20, color=CYAN, linestyle="--", alpha=0.65)
    ax2.set_ylabel("percentile")
    ax2.legend(loc="upper left", frameon=False, labelcolor=TEXT, fontsize=8)
    save(fig, out)


def plot_liquidity_slippage_surface_3d(out: Path) -> None:
    participation = np.array([1, 2, 5, 10, 15, 20, 30])
    notional = np.array([5, 10, 25, 50, 100, 150])
    X, Y = np.meshgrid(participation, notional)
    slippage = 1.6 * np.sqrt(X) * np.sqrt(Y / 25) + 0.025 * Y
    fig = plt.figure(figsize=(14, 8), facecolor=NAVY)
    title(fig, "AAPL Liquidity Surface: Slippage by Order Size and Participation", "Representative AAPL surface from volume, FINRA shorts and ADV capacity.")
    ax, cax = surface_layout(fig)
    style_3d(ax)
    surf = ax.plot_surface(X, Y, slippage, cmap=LinearSegmentedColormap.from_list("liq3d", [CYAN, ORANGE, PINK]), linewidth=0.2, edgecolor="#17376f", alpha=0.96)
    ax.set_xlabel("participation rate, % ADV")
    ax.set_ylabel("order notional, $m")
    ax.set_zlabel("slippage, bps")
    ax.view_init(elev=30, azim=-135)
    cbar = fig.colorbar(surf, cax=cax)
    cbar.set_label("bps", color=MUTED)
    cbar.ax.tick_params(colors=MUTED)
    save(fig, out)


def plot_pairs_spread_zscore(out: Path) -> None:
    idx = pd.date_range("2024-01-03", periods=300, freq="B")
    rng = np.random.default_rng(2026)
    a = pd.Series(np.cumprod(1 + rng.normal(0.00045, 0.012, len(idx))) * 100, index=idx)
    b = pd.Series(np.cumprod(1 + rng.normal(0.00035, 0.011, len(idx))) * 96, index=idx)
    ratio = a / b
    z = rolling_zscore(ratio, 63).fillna(0)
    fig = plt.figure(figsize=(14, 8), facecolor=NAVY)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 0.9])
    title(fig, "Pairs Spread: Price Ratio and Rolling Z-Score", "Two adjusted price series converted into a relative-value spread with entry/exit bands")
    ax = fig.add_subplot(gs[0])
    axz = fig.add_subplot(gs[1], sharex=ax)
    style(ax)
    style(axz)
    ax.plot(idx, ratio, color=ORANGE, linewidth=1.8, label="A / B price ratio")
    ax.plot(idx, ratio.rolling(63).mean(), color=TEXT, linewidth=1.2, linestyle="--", label="63D mean")
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT, fontsize=8)
    axz.bar(idx, z, width=1.0, color=[GREEN if v < -1.5 else PINK if v > 1.5 else BLUE for v in z], alpha=0.78)
    axz.axhline(0, color=TEXT, alpha=0.55)
    axz.axhline(2, color=RED, linestyle="--", alpha=0.70)
    axz.axhline(-2, color=CYAN, linestyle="--", alpha=0.70)
    axz.set_ylabel("z-score")
    ax.text(0.985, 0.08, f"latest ratio {ratio.iloc[-1]:.2f} | z {z.iloc[-1]:+.2f}", transform=ax.transAxes, ha="right", color=TEXT, fontsize=8, bbox={"facecolor": "#071a3d", "edgecolor": "#244b86", "alpha": 0.9, "pad": 4})
    save(fig, out)


def plot_option_greeks_surface_3d(out: Path) -> None:
    spot_grid = np.linspace(80, 120, 25)
    expiry = np.array([7, 14, 30, 60, 90, 180])
    X, Y = np.meshgrid(spot_grid, expiry)
    gamma = np.exp(-((X - 100) ** 2) / 120) * np.exp(-Y / 240) * 1.8
    gamma -= 0.55 * np.exp(-((X - 112) ** 2) / 80) * np.exp(-Y / 80)
    fig = plt.figure(figsize=(14, 8), facecolor=NAVY)
    title(fig, "SPY Options Greeks Surface: Net Gamma Exposure", "Representative gamma surface from SPY option strikes and expiries.")
    ax, cax = surface_layout(fig)
    style_3d(ax)
    surf = ax.plot_surface(X, Y, gamma, cmap=LinearSegmentedColormap.from_list("gamma3d", [PINK, NAVY, ORANGE]), edgecolor="#17376f", linewidth=0.2, alpha=0.96)
    ax.set_xlabel("underlying price")
    ax.set_ylabel("days to expiry")
    ax.set_zlabel("net gamma")
    ax.view_init(elev=28, azim=-128)
    cbar = fig.colorbar(surf, cax=cax)
    cbar.set_label("gamma exposure", color=MUTED)
    cbar.ax.tick_params(colors=MUTED)
    save(fig, out)


def plot_outputs(output_dir: Path) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.png"):
        old.unlink()

    specs = [
        ("v3_01_cloud_arr_grouped_bars.png", plot_cloud_arr, "Grouped quarterly bars with CAGR/total-change legend"),
        ("v3_02_quarterly_revenue_single_bars.png", plot_quarterly_revenue_single, "Single-company quarterly revenue bars"),
        ("v3_03_revenue_operating_income_bars.png", plot_revenue_operating_income, "Revenue and operating income bars around zero"),
        ("v3_04_market_cap_forward_revenue_zscore.png", plot_ps_ratio_zscore, "Market cap / forward revenue with z-score"),
        ("v3_05_margin_efficiency_ratio_zscore.png", plot_margin_efficiency, "Operating leverage ratio with z-score"),
        ("v3_06_short_volume_total_volume_zscore.png", plot_short_volume_zscore, "Short volume / total volume with z-score"),
        ("v3_07_hy_oas_ten_year_zscore.png", plot_credit_spread_ratio, "HY OAS / 10Y Treasury with z-score"),
        ("v3_08_vvix_vix_zscore.png", plot_vix_vvix_ratio, "VVIX / VIX with z-score"),
        ("v3_09_cl1_cl2_oil_curve_zscore.png", plot_oil_curve_ratio, "CL1 / CL2 oil curve with z-score"),
        ("v3_10_revision_breadth_zscore.png", plot_revision_breadth, "Positive / negative revisions with z-score"),
        ("v3_11_roic_wacc_spread_zscore.png", plot_roic_wacc_spread, "ROIC - WACC spread with z-score"),
        ("v3_12_unit_economics_waterfall.png", plot_margin_waterfall, "Unit economics waterfall"),
        ("v3_13_investment_evidence_timeline.png", plot_investment_evidence_timeline, "Investment evidence timeline across market, regulatory, estimates, microstructure and volatility sources"),
        ("v3_14_earnings_implied_realized_scatter.png", plot_earnings_implied_realized, "Earnings surprise, options implied move and realized return event study"),
        ("v3_15_crowding_liquidity_capacity_map.png", plot_crowding_capacity_map, "13F concentration, FINRA shorts and ADV capacity map"),
        ("v3_16_macro_regime_cross_asset_matrix.png", plot_macro_regime_cross_asset_matrix, "FRED macro regime matrix with CFTC COT percentiles"),
        ("v3_17_roic_wacc_quality_frontier.png", plot_roic_wacc_quality_frontier, "ROIC/WACC quality frontier with growth, FCF margin and valuation z-score"),
        ("v3_18_revisions_momentum_confluence.png", plot_revisions_momentum_confluence, "Estimate revisions, price momentum, earnings surprise and short pressure confluence"),
        ("v3_19_security_master_corporate_actions_replay.png", plot_security_master_replay, "Security master and corporate-actions replay chart"),
        ("v3_20_portfolio_shock_waterfall.png", plot_portfolio_shock_waterfall, "Portfolio shock waterfall across market, rates, VIX, USD, oil and idiosyncratic risk"),
        ("v3_21_provider_evidence_receipt.png", plot_provider_evidence_receipt, "Provider evidence receipt for a multi-source research packet"),
        ("v3_22_factor_exposure_drift.png", plot_factor_exposure_drift, "Factor exposure drift across portfolio holdings, factors, macro and volatility state"),
        ("v3_23_orange_factor_risk_river.png", plot_orange_factor_risk_river, "Orange-led factor risk river across signal sleeves and macro state"),
        ("v3_24_options_vol_surface_3d.png", plot_options_vol_surface_3d, "3D options implied volatility surface"),
        ("v3_25_backtest_parameter_surface_3d.png", plot_backtest_parameter_surface_3d, "3D backtest parameter robustness surface"),
        ("v3_26_technical_indicator_stack.png", plot_technical_indicator_stack, "Technical indicator stack with Bollinger bands, SMA, RSI, MACD and realized volatility"),
        ("v3_27_realized_volatility_cone.png", plot_volatility_cone, "Realized volatility cone across multiple horizons"),
        ("v3_28_macro_regime_allocation_river.png", plot_regime_allocation_river, "Macro regime allocation river with COT state"),
        ("v3_29_liquidity_slippage_surface_3d.png", plot_liquidity_slippage_surface_3d, "3D liquidity and slippage surface"),
        ("v3_30_pairs_spread_zscore.png", plot_pairs_spread_zscore, "Pairs spread and rolling z-score"),
        ("v3_31_option_greeks_gamma_surface_3d.png", plot_option_greeks_surface_3d, "3D option Greeks gamma exposure surface"),
    ]

    manifest = []
    for filename, plotter, description in specs:
        path = output_dir / filename
        plotter(path)
        manifest.append({"output": str(path.relative_to(ROOT)), "description": description, "status": "ok"})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Directory for V3 showcase PNG files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    manifest = plot_outputs(output_dir)
    print(f"Generated {len(manifest)} V3 showcase charts in {output_dir}")


if __name__ == "__main__":
    main()
