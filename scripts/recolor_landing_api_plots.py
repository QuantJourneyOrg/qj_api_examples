from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TARGET_BG = np.array([5, 25, 70], dtype=np.uint8)

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


def recolor(path: Path) -> int:
    image = Image.open(path).convert("RGBA")
    pixels = np.array(image)
    rgb = pixels[:, :, :3]
    alpha = pixels[:, :, 3]

    # Replace only very dark blue/black chart backgrounds and panel fills.
    mask = (
        (alpha > 0)
        & (rgb[:, :, 0] <= 12)
        & (rgb[:, :, 1] <= 32)
        & (rgb[:, :, 2] <= 90)
        & (rgb[:, :, 2] >= rgb[:, :, 0])
    )
    pixels[:, :, :3][mask] = TARGET_BG
    Image.fromarray(pixels, "RGBA").save(path)
    return int(mask.sum())


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in PLOTS if not path.exists()]
    if missing:
        raise SystemExit("Missing plots:\n" + "\n".join(missing))

    for path in PLOTS:
        changed = recolor(path)
        print(f"{path.relative_to(ROOT)}: recolored {changed:,} pixels")


if __name__ == "__main__":
    main()
