from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TARGET = np.array([168, 85, 247], dtype=np.float32) / 255.0
TARGET_HSV = None

PLOTS = [
    ROOT / "_output" / "_v3" / "v3_01_cloud_arr_grouped_bars.png",
    ROOT / "_output" / "_v3" / "v3_14_earnings_implied_realized_scatter.png",
    ROOT / "_output" / "_v3" / "v3_23_orange_factor_risk_river.png",
    ROOT / "_output" / "_v3" / "v3_25_backtest_parameter_surface_3d.png",
    ROOT / "_output" / "_v3" / "v3_27_realized_volatility_cone.png",
    ROOT / "_output" / "_v3" / "v3_30_pairs_spread_zscore.png",
    ROOT / "_output" / "_v3" / "v3_11_roic_wacc_spread_zscore.png",
    ROOT / "_output" / "39_institutional_crowding_13f_flows_output_01.png",
]

YELLOW_GREEN_PLOTS = {
    ROOT / "_output" / "39_institutional_crowding_13f_flows_output_01.png",
}


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


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6).astype(np.int32)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6

    out = np.zeros(hsv.shape, dtype=np.float32)
    choices = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ]
    for idx, vals in enumerate(choices):
        mask = i == idx
        out[..., 0][mask], out[..., 1][mask], out[..., 2][mask] = vals[0][mask], vals[1][mask], vals[2][mask]
    return out


def recolor(path: Path) -> int:
    image = Image.open(path).convert("RGBA")
    pixels = np.array(image)
    rgb = pixels[..., :3].astype(np.float32) / 255.0
    alpha = pixels[..., 3]
    hsv = rgb_to_hsv(rgb)

    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    warm_yellow_orange = (
        (alpha > 0)
        & (hue >= 0.055)
        & (hue <= 0.185)
        & (sat >= 0.28)
        & (val >= 0.42)
        & (rgb[..., 0] >= 0.45)
        & (rgb[..., 1] >= 0.28)
        & (rgb[..., 2] <= 0.72)
    )
    if path in YELLOW_GREEN_PLOTS:
        warm_yellow_orange = warm_yellow_orange | (
            (alpha > 0)
            & (hue >= 0.15)
            & (hue <= 0.30)
            & (sat >= 0.22)
            & (val >= 0.45)
            & (rgb[..., 0] >= 0.45)
            & (rgb[..., 1] >= 0.45)
            & (rgb[..., 2] <= 0.75)
        )

    target_hue = rgb_to_hsv(TARGET.reshape(1, 1, 3))[0, 0, 0]
    hsv[..., 0][warm_yellow_orange] = target_hue
    hsv[..., 1][warm_yellow_orange] = np.clip(hsv[..., 1][warm_yellow_orange] * 0.92, 0.38, 0.92)
    recolored = np.clip(hsv_to_rgb(hsv) * 255, 0, 255).astype(np.uint8)
    pixels[..., :3][warm_yellow_orange] = recolored[warm_yellow_orange]
    Image.fromarray(pixels, "RGBA").save(path)
    return int(warm_yellow_orange.sum())


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in PLOTS if not path.exists()]
    if missing:
        raise SystemExit("Missing plots:\n" + "\n".join(missing))
    for path in PLOTS:
        changed = recolor(path)
        print(f"{path.relative_to(ROOT)}: recolored {changed:,} pixels")


if __name__ == "__main__":
    main()
