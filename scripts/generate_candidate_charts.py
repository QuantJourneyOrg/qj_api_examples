"""Generate one dark preview chart for every notebook in `_candidates/`.

These are visual previews for the candidate catalog, not notebook execution
artifacts. Each output is a single-axes PNG so the catalog never shows two
separate charts inside the same image.
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
OUTPUT = ROOT / "outputs" / "candidates"

NAVY = "#051946"
PANEL = "#071f55"
TEXT = "#eaf1ff"
MUTED = "#9fb1d1"
BLUE = "#4da3ff"
CYAN = "#58d5ff"
PINK = "#ff6fb3"
GREEN = "#66e0a3"
AMBER = "#ffc857"
RED = "#ff7a7a"
CORR_CMAP = LinearSegmentedColormap.from_list("qj_corr", [CYAN, PANEL, PINK])


def rng_for(stem: str) -> np.random.Generator:
    seed = int(hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def style_ax(ax: plt.Axes, title: str) -> None:
    fig = ax.figure
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(PANEL)
    ax.set_title(title.replace("_", " "), color=TEXT, fontsize=16, pad=14, weight="medium")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color="white", alpha=0.10, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#17376f")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def save(fig: plt.Figure, path: Path) -> None:
    if len(fig.axes) != 1:
        raise RuntimeError(f"{path.name} has {len(fig.axes)} axes; expected exactly one")
    fig.tight_layout(pad=1.6)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def dates(n: int = 252) -> pd.DatetimeIndex:
    return pd.bdate_range(end="2026-06-05", periods=n)


def random_walk(rng: np.random.Generator, n: int = 252, drift: float = 0.0004, vol: float = 0.012) -> pd.Series:
    return pd.Series(np.cumprod(1 + rng.normal(drift, vol, n)), index=dates(n))


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


def plot_one(stem: str, out: Path) -> None:
    title = stem[3:] if stem[:2].isdigit() else stem
    lower = stem.lower()
    if "market_data" in lower or "pricing" in lower:
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
    parser.add_argument(
        "--output-style",
        choices=["preview", "run"],
        default="preview",
        help="preview writes <stem>.png; run writes <stem>_output_01.png.",
    )
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
    print(f"Generated {len(notebooks)} candidate charts in {output_dir}")


if __name__ == "__main__":
    main()
