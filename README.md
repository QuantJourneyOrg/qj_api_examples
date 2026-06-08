# QuantJourney API Examples

Executed, sanitized notebooks for the QuantJourney API and SDK.

This repository shows practical examples for financial data access, direct SDK connector calls, analytics outputs, governed workflow patterns, and buy-side research. The notebooks use `QJ_API_KEY` from the environment and do not include vendor credentials or hardcoded QuantJourney tokens.

Documentation: https://api.quantjourney.cloud/docs

API landing: https://api.quantjourney.cloud/_v3

Examples catalog: https://api.quantjourney.cloud/examples

SDK: `pip install quantjourney`

## Quick Start

```python
from quantjourney.sdk import QuantJourneyAPI

qj = QuantJourneyAPI(api_key="qj_...")

prices = qj.eod.get_historical_prices(
    symbol="AAPL",
    start_date="2024-01-01",
    end_date="2024-12-31",
)
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install quantjourney pandas numpy matplotlib plotly nbformat nbclient ipykernel

export QJ_API_KEY="qj_..."
jupyter lab
```

To re-run one notebook:

```bash
jupyter nbconvert --execute --to notebook --inplace notebooks/core/02_market_data_basics.ipynb
```

To regenerate the six landing-page PNG outputs from live API data:

```bash
export QJ_API_KEY="qj_..."
PYTHONPATH=/path/to/quantjourney-sdk python scripts/generate_landing_outputs.py
```

To regenerate the advanced buy-side PNG outputs from live API data:

```bash
export QJ_API_KEY="qj_..."
PYTHONPATH=/path/to/quantjourney-sdk python scripts/generate_advanced_buy_side_outputs.py
```

## Landing Recipes

These are the six notebook outputs used by the QuantJourney API landing page.

| Workflow | Domain | Notebook | Output |
|---|---|---|---|
| Market data | `/d/equity/pricing` | [02_market_data_basics](notebooks/core/02_market_data_basics.ipynb) | [PNG](outputs/landing/recipe-market-data-real.png) |
| Macro dashboard | `/d/macro/rates`, `/d/macro/series` | [03_economic_data_macro](notebooks/core/03_economic_data_macro.ipynb) | [PNG](outputs/landing/recipe-macro-dashboard-real.png) |
| Peer valuation | `/d/equity/fundamentals` | [04_fundamental_analysis](notebooks/core/04_fundamental_analysis.ipynb) | [PNG](outputs/landing/recipe-peer-valuation-real.png) |
| Portfolio risk snapshot | `/d/portfolio/analytics` | [06_portfolio_analysis](notebooks/core/06_portfolio_analysis.ipynb) | [PNG](outputs/landing/recipe-portfolio-risk-real.png) |
| VIX / volatility regime | `/d/derivatives/vol` | [08_cboe_vix](notebooks/core/08_cboe_vix.ipynb) | [PNG](outputs/landing/recipe-vix-regime-real.png) |
| Risk parity / allocation | `/d/optimizer/portfolio` | [26_risk_parity_portfolio](notebooks/buy_side/26_risk_parity_portfolio.ipynb) | [PNG](outputs/landing/recipe-risk-parity-real.png) |

## Advanced Buy-Side Labs

These examples use live QuantJourney price data and vectorized pandas/numpy diagnostics for workflows that portfolio teams usually ask for after basic data access is solved: parameter sensitivity, walk-forward validation, tail-risk simulation, drawdown state, correlation regimes and rolling factor exposure.

| Notebook | Workflow | Output |
|---|---|---|
| [61_vectorized_strategy_grid](notebooks/buy_side_advanced/61_vectorized_strategy_grid.ipynb) | SMA parameter sweep with transaction costs and Sharpe heatmap. | [PNG](outputs/buy_side_advanced/advanced-sma-grid-heatmap.png) |
| [62_walk_forward_robustness](notebooks/buy_side_advanced/62_walk_forward_robustness.ipynb) | Rolling train/test parameter selection and out-of-sample equity. | [PNG](outputs/buy_side_advanced/advanced-walk-forward-matrix.png) |
| [63_monte_carlo_tail_risk](notebooks/buy_side_advanced/63_monte_carlo_tail_risk.ipynb) | Bootstrap fan chart and terminal-return distribution for a portfolio basket. | [PNG](outputs/buy_side_advanced/advanced-monte-carlo-tail-risk.png) |
| [64_correlation_regime_lab](notebooks/buy_side_advanced/64_correlation_regime_lab.ipynb) | Cross-asset correlation heatmap, rolling correlation and drawdown context. | [PNG](outputs/buy_side_advanced/advanced-correlation-regime-map.png) |
| [65_drawdown_diagnostics](notebooks/buy_side_advanced/65_drawdown_diagnostics.ipynb) | Underwater drawdown, rolling volatility and strategy exposure state. | [PNG](outputs/buy_side_advanced/advanced-drawdown-diagnostics.png) |
| [66_factor_exposure_diagnostics](notebooks/buy_side_advanced/66_factor_exposure_diagnostics.ipynb) | Rolling 126D factor betas and recent factor contribution proxy. | [PNG](outputs/buy_side_advanced/advanced-factor-exposure-diagnostics.png) |

## Buy-Side Candidate Catalog

The flat [`_candidates`](./_candidates/INDEX.md) catalog contains core notebooks `01-13` plus clean institutional workflow candidates for construction, liquidity, capacity, stress, attribution, regulatory signals, options overlays and end-to-end PM reporting. Candidate notebooks are source files with direct SDK calls; generated plot artifacts live in [`plots`](./plots) and are indexed by [`plots/manifest.json`](./plots/manifest.json). Candidate notebooks use SDK calls such as `qj.eod`, `qj.fmp`, `qj.sec`, `qj.cboe`, `qj.cftc`, `qj.ff`, `qj.bt` and `qj.analytics`, then keep portfolio/risk calculations visible in pandas/numpy. Production systems can wrap the same workflows behind governed domain routes, tenant scopes and audit metadata.

Notebooks `11-19` focus on core institutional data primitives: SEC filings, FINRA short interest, OpenFIGI identity, adjustment semantics, domain route discovery, global macro sources, volatility feeds, index universe construction and lineage/audit packets.

Notebooks `80-88` focus on multi-source institutional workflows: evidence packets, pre-trade capacity, PEAD research, PM risk briefs, crowding flows, cross-asset macro scenarios, macro regime allocation, inflation shocks and COT positioning.

To rebuild the catalog:

```bash
export QJ_API_KEY="qj_..."
PYTHONPATH=/path/to/quantjourney-sdk python scripts/build_candidate_notebooks.py
python scripts/generate_candidate_charts.py
python scripts/run_candidate_files.py
```

`scripts/run_candidate_files.py` processes every `_candidates/*.ipynb` file and writes one chart per file to `plots/<notebook>_output_01.png`, plus `plots/manifest.json`.

## Core Notebooks

| # | Notebook | What It Shows | Outputs |
|---|---|---|---|
| 01 | [Authentication Methods](notebooks/core/01_authentication_methods.ipynb) | API key setup, environment variables, auth patterns, connection testing. | Embedded text output |
| 02 | [Market Data Basics](notebooks/core/02_market_data_basics.ipynb) | Historical OHLCV, candlestick chart, multi-symbol comparison, returns, volatility, correlation. | [output folder](outputs/core/02_market_data_basics) |
| 03 | [Economic Data and Macro](notebooks/core/03_economic_data_macro.ipynb) | GDP, CPI, unemployment, Treasury yields, Fed Funds and macro dashboard. | [output folder](outputs/core/03_economic_data_macro) |
| 04 | [Fundamental Analysis](notebooks/core/04_fundamental_analysis.ipynb) | Financial statements, ratios, margins, peer valuation and company comparison. | [output folder](outputs/core/04_fundamental_analysis) |
| 05 | [Technical Analysis](notebooks/core/05_technical_analysis.ipynb) | SMA, Bollinger bands, RSI, MACD, rolling volatility and risk metrics. | [output folder](outputs/core/05_technical_analysis) |
| 06 | [Portfolio Analysis](notebooks/core/06_portfolio_analysis.ipynb) | Portfolio returns, correlation, drawdown, covariance and risk contribution. | [output folder](outputs/core/06_portfolio_analysis) |
| 07 | [Crypto CCXT](notebooks/core/07_crypto_ccxt.ipynb) | Crypto exchange data, symbols, OHLCV, order books and funding examples. | Embedded text output |
| 08 | [CBOE VIX](notebooks/core/08_cboe_vix.ipynb) | VIX history, fear regimes, moving averages and volatility context. | [output folder](outputs/core/08_cboe_vix) |
| 09 | [Multpl Valuation](notebooks/core/09_multpl_valuation.ipynb) | Shiller P/E, dividend yield and market valuation context. | [output folder](outputs/core/09_multpl_valuation) |
| 10 | [CFTC COT](notebooks/core/10_cftc_cot.ipynb) | Commitment of Traders positioning and futures sentiment. | [output folder](outputs/core/10_cftc_cot) |
| 11 | [SEC Filings](notebooks/core/11_sec_filings.ipynb) | Company filings, disclosure search and filing metadata. | Embedded text output |
| 12 | [FINRA Short Interest](notebooks/core/12_finra_short_interest.ipynb) | Short interest, short-volume context and crowding indicators. | Embedded text output |
| 13 | [OpenFIGI](notebooks/core/13_openfigi.ipynb) | FIGI lookup, symbol mapping and security identity. | Embedded text output |

## Buy-Side Examples

These examples are closer to hedge fund, family-office and institutional research workflows.

| Notebook | Workflow | Outputs |
|---|---|---|
| [20_multi_factor_model](notebooks/buy_side/20_multi_factor_model.ipynb) | Multi-factor exposures, model inputs and factor diagnostics. | [output folder](outputs/buy_side/20_multi_factor_model) |
| [21_volatility_surface_greeks](notebooks/buy_side/21_volatility_surface_greeks.ipynb) | Options surface, implied volatility and Greeks review. | [output folder](outputs/buy_side/21_volatility_surface_greeks) |
| [23_cot_positioning_sentiment](notebooks/buy_side/23_cot_positioning_sentiment.ipynb) | Futures positioning, crowded exposure and sentiment context. | [output folder](outputs/buy_side/23_cot_positioning_sentiment) |
| [24_macro_regime_allocation](notebooks/buy_side/24_macro_regime_allocation.ipynb) | Regime classification, macro allocation and performance comparison. | [output folder](outputs/buy_side/24_macro_regime_allocation) |
| [25_cross_asset_correlation](notebooks/buy_side/25_cross_asset_correlation.ipynb) | Cross-asset correlations, clustering, stress correlations and PCA. | [output folder](outputs/buy_side/25_cross_asset_correlation) |
| [26_risk_parity_portfolio](notebooks/buy_side/26_risk_parity_portfolio.ipynb) | Risk parity weights, contribution, volatility targeting and performance. | [output folder](outputs/buy_side/26_risk_parity_portfolio) |
| [27_var_expected_shortfall](notebooks/buy_side/27_var_expected_shortfall.ipynb) | VaR, expected shortfall, drawdowns and tail-risk diagnostics. | [output folder](outputs/buy_side/27_var_expected_shortfall) |
| [28_factor_attribution](notebooks/buy_side/28_factor_attribution.ipynb) | Factor attribution, exposure diagnostics and contribution analysis. | [output folder](outputs/buy_side/28_factor_attribution) |
| [29_pairs_trading_stat_arb](notebooks/buy_side/29_pairs_trading_stat_arb.ipynb) | Pair selection, cointegration, spread z-score and stat-arb diagnostics. | [output folder](outputs/buy_side/29_pairs_trading_stat_arb) |
| [31_event_study_earnings](notebooks/buy_side/31_event_study_earnings.ipynb) | Earnings event study, abnormal returns and event-window analysis. | [output folder](outputs/buy_side/31_event_study_earnings) |
| [35_performance_reporting](notebooks/buy_side/35_performance_reporting.ipynb) | Report-ready performance, drawdown, rolling metrics and contribution views. | [output folder](outputs/buy_side/35_performance_reporting) |
| [40_sector_rotation_momentum](notebooks/buy_side/40_sector_rotation_momentum.ipynb) | Sector momentum, rotation signals and allocation context. | [output folder](outputs/buy_side/40_sector_rotation_momentum) |
| [43_tail_risk_hedging](notebooks/buy_side/43_tail_risk_hedging.ipynb) | Tail-risk hedging, stress periods and hedge payoff context. | [output folder](outputs/buy_side/43_tail_risk_hedging) |
| [51_factor_risk_attribution](notebooks/buy_side/51_factor_risk_attribution.ipynb) | Factor risk decomposition, sector/style contributors and risk review. | [output folder](outputs/buy_side/51_factor_risk_attribution) |
| [59_risk_adjusted_performance](notebooks/buy_side/59_risk_adjusted_performance.ipynb) | Sharpe, Sortino, Calmar, drawdown and risk-adjusted comparison. | [output folder](outputs/buy_side/59_risk_adjusted_performance) |

## Output Manifest

`outputs/manifest.json` records the generated notebooks and extracted PNG counts. Notebooks without PNG charts still retain their executed table/text output inside the notebook.

The six files in `outputs/landing/` are generated from live QuantJourney API calls by `scripts/generate_landing_outputs.py`.

## Data and Licensing

QuantJourney is a software and API control-plane layer. Market data access depends on your own QuantJourney tenant configuration and any required provider licenses or BYOL credentials.
