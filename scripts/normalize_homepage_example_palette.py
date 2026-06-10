from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TARGET_BG = np.array([5, 25, 70], dtype=np.uint8)
CYAN = np.array([88, 213, 255], dtype=np.float32) / 255.0
PINK = np.array([192, 68, 134], dtype=np.float32) / 255.0
PURPLE = np.array([168, 85, 247], dtype=np.float32) / 255.0

PLOTS = [
    ROOT / "_output" / "_v3" / "v3_01_cloud_arr_grouped_bars.png",
    ROOT / "_output" / "_v3" / "v3_14_earnings_implied_realized_scatter.png",
    ROOT / "_output" / "_v3" / "v3_16_macro_regime_cross_asset_matrix.png",
    ROOT / "_output" / "_v3" / "v3_23_orange_factor_risk_river.png",
    ROOT / "_output" / "_v3" / "v3_25_backtest_parameter_surface_3d.png",
    ROOT / "_output" / "_v3" / "v3_26_technical_indicator_stack.png",
    ROOT / "_output" / "_v3" / "v3_27_realized_volatility_cone.png",
    ROOT / "_output" / "_v3" / "v3_30_pairs_spread_zscore.png",
    ROOT / "_output" / "32_liquidity_capacity_impact_output_01.png",
    ROOT / "_output" / "39_institutional_crowding_13f_flows_output_01.png",
    ROOT / "_output" / "55_factor_timing_dynamic_exposures_output_01.png",
    ROOT / "_output" / "63_monte_carlo_tail_risk_output_01.png",
    ROOT / "_output" / "73_conditioned_earnings_pead_multi_source_output_01.png",
    ROOT / "_output" / "02_market_01.png",
    ROOT / "_output" / "_v3" / "v3_10_revision_breadth_zscore.png",
    ROOT / "_output" / "_v3" / "v3_11_roic_wacc_spread_zscore.png",
    ROOT / "_output" / "_v3" / "v3_31_option_greeks_gamma_surface_3d.png",
]


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    nonzero = delta > 1e-6
    rmax = nonzero & (maxc == r)
    gmax = nonzero & (maxc == g)
    bmax = nonzero & (maxc == b)
    hue[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6
    hue[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2
    hue[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4
    hue /= 6.0

    sat = np.zeros_like(maxc)
    sat[maxc > 1e-6] = delta[maxc > 1e-6] / maxc[maxc > 1e-6]
    return np.stack([hue, sat, maxc], axis=-1)


def apply_tint(rgb: np.ndarray, mask: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    if not mask.any():
        return out
    hsv = rgb_to_hsv(rgb)
    value = hsv[..., 2][mask]
    sat = np.clip(hsv[..., 1][mask], 0.38, 0.92)
    target_mix = target.reshape(1, 3)
    tinted = target_mix * (0.52 + value[:, None] * 0.62)
    neutral = value[:, None] * np.array([0.86, 0.92, 1.0], dtype=np.float32).reshape(1, 3)
    out[mask] = np.clip(tinted * sat[:, None] + neutral * (1 - sat[:, None]) * 0.36, 0, 1)
    return out


def normalize(path: Path) -> dict[str, int]:
    image = Image.open(path).convert("RGBA")
    pixels = np.array(image)
    rgb_u8 = pixels[..., :3]
    alpha = pixels[..., 3]
    rgb = rgb_u8.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(rgb)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    dark_bg = (
        (alpha > 0)
        & (rgb_u8[..., 0] <= 14)
        & (rgb_u8[..., 1] <= 35)
        & (rgb_u8[..., 2] <= 96)
        & (rgb_u8[..., 2] >= rgb_u8[..., 0])
    )
    pixels[..., :3][dark_bg] = TARGET_BG

    warm = (
        (alpha > 0)
        & (hue >= 0.035)
        & (hue <= 0.185)
        & (sat >= 0.24)
        & (val >= 0.35)
    )
    hard_orange = (
        (alpha > 0)
        & (rgb[..., 0] > 0.58)
        & (rgb[..., 1] > 0.16)
        & (rgb[..., 1] < 0.64)
        & (rgb[..., 2] < 0.34)
    )
    hard_yellow = (
        (alpha > 0)
        & (rgb[..., 0] > 0.62)
        & (rgb[..., 1] > 0.54)
        & (rgb[..., 2] < 0.32)
    )
    yellow_green = (
        (alpha > 0)
        & (hue > 0.185)
        & (hue <= 0.33)
        & (sat >= 0.20)
        & (val >= 0.38)
    )
    green = (
        (alpha > 0)
        & (hue > 0.33)
        & (hue <= 0.47)
        & (sat >= 0.20)
        & (val >= 0.35)
    )
    red = (
        (alpha > 0)
        & ((hue <= 0.025) | (hue >= 0.96))
        & (sat >= 0.26)
        & (val >= 0.35)
    )

    rgb = pixels[..., :3].astype(np.float32) / 255.0
    rgb = apply_tint(rgb, warm | red | hard_orange, PINK)
    rgb = apply_tint(rgb, yellow_green | hard_yellow, PURPLE)
    rgb = apply_tint(rgb, green, CYAN)
    pixels[..., :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, "RGBA").save(path)
    return {
        "background": int(dark_bg.sum()),
        "warm_red": int((warm | red | hard_orange).sum()),
        "yellow_green": int((yellow_green | hard_yellow).sum()),
        "green": int(green.sum()),
    }


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in PLOTS if not path.exists()]
    if missing:
        raise SystemExit("Missing plots:\n" + "\n".join(missing))

    for path in PLOTS:
        stats = normalize(path)
        stat_text = ", ".join(f"{key}={value:,}" for key, value in stats.items())
        print(f"{path.relative_to(ROOT)}: {stat_text}")


if __name__ == "__main__":
    main()
