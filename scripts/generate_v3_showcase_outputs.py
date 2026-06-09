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


def title(fig: plt.Figure, heading: str, subtitle: str) -> None:
    fig.patch.set_facecolor(NAVY)
    fig.text(0.035, 0.965, heading, color=TEXT, fontsize=22, weight="semibold", va="top")
    fig.text(0.035, 0.915, subtitle, color=MUTED, fontsize=10.5, va="top")


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
