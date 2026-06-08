# Buy-Side Candidate Notebook Catalog

Flat candidate catalog for institutional workflows. Candidate notebooks are clean source notebooks; generated run plots are committed under `plots/` and indexed in `plots/manifest.json`. New candidates use real QuantJourney SDK connector calls plus transparent local analytics. Production systems can wrap the same workflows behind governed domain routes, tenant scopes and audit metadata.

| Notebook | Preview | Category | What it shows |
|---|---|---|---|
| [01_authentication_methods.ipynb](01_authentication_methods.ipynb) | [PNG](../plots/01_authentication_methods_output_01.png) | Existing core notebook | copied from notebooks/core |
| [02_market_data_basics.ipynb](02_market_data_basics.ipynb) | [PNG](../plots/02_market_data_basics_output_01.png) | Existing core notebook | copied from notebooks/core |
| [03_economic_data_macro.ipynb](03_economic_data_macro.ipynb) | [PNG](../plots/03_economic_data_macro_output_01.png) | Existing core notebook | copied from notebooks/core |
| [04_fundamental_analysis.ipynb](04_fundamental_analysis.ipynb) | [PNG](../plots/04_fundamental_analysis_output_01.png) | Existing core notebook | copied from notebooks/core |
| [05_technical_analysis.ipynb](05_technical_analysis.ipynb) | [PNG](../plots/05_technical_analysis_output_01.png) | Existing core notebook | copied from notebooks/core |
| [06_portfolio_analysis.ipynb](06_portfolio_analysis.ipynb) | [PNG](../plots/06_portfolio_analysis_output_01.png) | Existing core notebook | copied from notebooks/core |
| [07_crypto_ccxt.ipynb](07_crypto_ccxt.ipynb) | [PNG](../plots/07_crypto_ccxt_output_01.png) | Existing core notebook | copied from notebooks/core |
| [08_cboe_vix.ipynb](08_cboe_vix.ipynb) | [PNG](../plots/08_cboe_vix_output_01.png) | Existing core notebook | copied from notebooks/core |
| [09_multpl_valuation.ipynb](09_multpl_valuation.ipynb) | [PNG](../plots/09_multpl_valuation_output_01.png) | Existing core notebook | copied from notebooks/core |
| [10_cftc_cot.ipynb](10_cftc_cot.ipynb) | [PNG](../plots/10_cftc_cot_output_01.png) | Existing core notebook | copied from notebooks/core |
| [20_multi_factor_model.ipynb](20_multi_factor_model.ipynb) | [PNG](../plots/20_multi_factor_model_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [21_volatility_surface_greeks.ipynb](21_volatility_surface_greeks.ipynb) | [PNG](../plots/21_volatility_surface_greeks_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [22_hierarchical_risk_parity_hrp.ipynb](22_hierarchical_risk_parity_hrp.ipynb) | [PNG](../plots/22_hierarchical_risk_parity_hrp_output_01.png) | Portfolio construction | Compares HRP-style recursive allocation, inverse-volatility risk parity and minimum variance using the same price panel. |
| [23_cot_positioning_sentiment.ipynb](23_cot_positioning_sentiment.ipynb) | [PNG](../plots/23_cot_positioning_sentiment_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [24_macro_regime_allocation.ipynb](24_macro_regime_allocation.ipynb) | [PNG](../plots/24_macro_regime_allocation_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [25_cross_asset_correlation.ipynb](25_cross_asset_correlation.ipynb) | [PNG](../plots/25_cross_asset_correlation_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [26_risk_parity_portfolio.ipynb](26_risk_parity_portfolio.ipynb) | [PNG](../plots/26_risk_parity_portfolio_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [27_var_expected_shortfall.ipynb](27_var_expected_shortfall.ipynb) | [PNG](../plots/27_var_expected_shortfall_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [28_factor_attribution.ipynb](28_factor_attribution.ipynb) | [PNG](../plots/28_factor_attribution_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [29_pairs_trading_stat_arb.ipynb](29_pairs_trading_stat_arb.ipynb) | [PNG](../plots/29_pairs_trading_stat_arb_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [30_universe_construction_liquidity_screen.ipynb](30_universe_construction_liquidity_screen.ipynb) | [PNG](../plots/30_universe_construction_liquidity_screen_output_01.png) | Universe construction | Builds an investable equity universe from prices, volumes, optional FMP screener output, TTM ratios and short-interest context. |
| [31_event_study_earnings.ipynb](31_event_study_earnings.ipynb) | [PNG](../plots/31_event_study_earnings_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [32_liquidity_capacity_impact.ipynb](32_liquidity_capacity_impact.ipynb) | [PNG](../plots/32_liquidity_capacity_impact_output_01.png) | Liquidity and capacity | Estimates position capacity, participation limits and simple Amihud-style impact using adjusted prices and volume. |
| [33_stress_testing_macro_scenarios.ipynb](33_stress_testing_macro_scenarios.ipynb) | [PNG](../plots/33_stress_testing_macro_scenarios_output_01.png) | Risk and scenarios | Maps custom macro and market shocks onto a portfolio through observed factor betas. |
| [34_index_replication_tracking_error.ipynb](34_index_replication_tracking_error.ipynb) | [PNG](../plots/34_index_replication_tracking_error_output_01.png) | Portfolio construction | Builds a constrained replication basket and measures tracking error against SPY. |
| [35_performance_reporting.ipynb](35_performance_reporting.ipynb) | [PNG](../plots/35_performance_reporting_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [36_rolling_risk_budgeting_drawdown_control.ipynb](36_rolling_risk_budgeting_drawdown_control.ipynb) | [PNG](../plots/36_rolling_risk_budgeting_drawdown_control_output_01.png) | Risk management | Applies volatility targeting, rolling risk budgeting and drawdown exposure control to a multi-asset portfolio. |
| [37_brinson_attribution_benchmark_relative.ipynb](37_brinson_attribution_benchmark_relative.ipynb) | [PNG](../plots/37_brinson_attribution_benchmark_relative_output_01.png) | Attribution | Calculates allocation, selection and interaction effects using holdings, benchmark weights and sector return buckets. |
| [38_congress_smart_money_overlay.ipynb](38_congress_smart_money_overlay.ipynb) | [PNG](../plots/38_congress_smart_money_overlay_output_01.png) | Alternative data / signals | Combines congressional trade feeds with price reactions to create a transparent event overlay. |
| [39_institutional_crowding_13f_flows.ipynb](39_institutional_crowding_13f_flows.ipynb) | [PNG](../plots/39_institutional_crowding_13f_flows_output_01.png) | Regulatory / crowding | Uses SEC/FMP institutional ownership calls with returns data to build crowding, concentration and flow proxies. |
| [40_sector_rotation_momentum.ipynb](40_sector_rotation_momentum.ipynb) | [PNG](../plots/40_sector_rotation_momentum_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [41_broad_event_study_pead.ipynb](41_broad_event_study_pead.ipynb) | [PNG](../plots/41_broad_event_study_pead_output_01.png) | Event studies | Extends earnings analysis into a multi-name post-earnings drift workflow with event windows and sector splits. |
| [42_factor_risk_model_construction.ipynb](42_factor_risk_model_construction.ipynb) | [PNG](../plots/42_factor_risk_model_construction_output_01.png) | Risk models | Builds sample, shrinkage and factor-model covariance estimates from price and factor returns. |
| [43_tail_risk_hedging.ipynb](43_tail_risk_hedging.ipynb) | [PNG](../plots/43_tail_risk_hedging_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [44_cta_futures_carry_trend_macro.ipynb](44_cta_futures_carry_trend_macro.ipynb) | [PNG](../plots/44_cta_futures_carry_trend_macro_output_01.png) | CTA / futures | Creates a simple CTA-like signal book using trend proxies, optional COT positioning and macro context. |
| [45_holdings_based_vs_returns_based.ipynb](45_holdings_based_vs_returns_based.ipynb) | [PNG](../plots/45_holdings_based_vs_returns_based_output_01.png) | Attribution / ownership | Compares 13F/holdings-style concentration with returns-based factor exposure analysis. |
| [46_options_overlay_strategies.ipynb](46_options_overlay_strategies.ipynb) | [PNG](../plots/46_options_overlay_strategies_output_01.png) | Options / overlays | Uses option-chain, VIX/SKEW and underlying returns to compare covered-call, put-write and collar payoff profiles. |
| [47_tactical_asset_allocation_macro_valuation.ipynb](47_tactical_asset_allocation_macro_valuation.ipynb) | [PNG](../plots/47_tactical_asset_allocation_macro_valuation_output_01.png) | Asset allocation | Combines macro, valuation and sentiment feeds with market data to create a dynamic allocation rule. |
| [48_end_to_end_research_to_book.ipynb](48_end_to_end_research_to_book.ipynb) | [PNG](../plots/48_end_to_end_research_to_book_output_01.png) | End-to-end workflow | Runs universe scoring, weight construction, risk attribution, drawdown review and report-ready outputs in one notebook. |
| [50_integrated_daily_risk_attribution_report.ipynb](50_integrated_daily_risk_attribution_report.ipynb) | [PNG](../plots/50_integrated_daily_risk_attribution_report_output_01.png) | Daily PM report | Produces a daily intelligence packet from holdings, factor moves, earnings/filings/insider context and risk contributors. |
| [51_factor_risk_attribution.ipynb](51_factor_risk_attribution.ipynb) | [PNG](../plots/51_factor_risk_attribution_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [52_turnover_cost_drag.ipynb](52_turnover_cost_drag.ipynb) | [PNG](../plots/52_turnover_cost_drag_output_01.png) | Execution-aware backtesting | Quantifies how turnover, slippage assumptions and rebalance frequency change an otherwise attractive signal. |
| [53_backtest_with_risk_management_full.ipynb](53_backtest_with_risk_management_full.ipynb) | [PNG](../plots/53_backtest_with_risk_management_full_output_01.png) | Backtesting | Compares a raw momentum strategy with a risk-managed variant using volatility targeting, drawdown controls and concentration limits. |
| [55_factor_timing_dynamic_exposures.ipynb](55_factor_timing_dynamic_exposures.ipynb) | [PNG](../plots/55_factor_timing_dynamic_exposures_output_01.png) | Factor timing | Uses macro regimes and factor proxies to time equity, growth, small-cap, duration and gold exposures. |
| [57_crypto_perps_basis_funding_arbitrage.ipynb](57_crypto_perps_basis_funding_arbitrage.ipynb) | [PNG](../plots/57_crypto_perps_basis_funding_arbitrage_output_01.png) | Crypto / derivatives | Pulls CCXT-style spot/perp data and funding history where available, then builds funding and basis diagnostics. |
| [59_risk_adjusted_performance.ipynb](59_risk_adjusted_performance.ipynb) | [PNG](../plots/59_risk_adjusted_performance_output_01.png) | Existing executed buy-side notebook | copied from notebooks/buy_side |
| [61_vectorized_strategy_grid.ipynb](61_vectorized_strategy_grid.ipynb) | [PNG](../plots/61_vectorized_strategy_grid_output_01.png) | Advanced buy-side diagnostics | Runs a complete SMA parameter sweep from live QuantJourney price data, including transaction costs and robustness surfaces. |
| [62_walk_forward_robustness.ipynb](62_walk_forward_robustness.ipynb) | [PNG](../plots/62_walk_forward_robustness_output_01.png) | Advanced buy-side diagnostics | Selects parameters on rolling training windows and evaluates the next out-of-sample fold. |
| [63_monte_carlo_tail_risk.ipynb](63_monte_carlo_tail_risk.ipynb) | [PNG](../plots/63_monte_carlo_tail_risk_output_01.png) | Advanced buy-side diagnostics | Bootstraps portfolio returns into one-year fan charts and terminal-return tail distributions. |
| [64_correlation_regime_lab.ipynb](64_correlation_regime_lab.ipynb) | [PNG](../plots/64_correlation_regime_lab_output_01.png) | Advanced buy-side diagnostics | Calculates recent cross-asset correlation, rolling pairwise regimes and drawdown context. |
| [65_drawdown_diagnostics.ipynb](65_drawdown_diagnostics.ipynb) | [PNG](../plots/65_drawdown_diagnostics_output_01.png) | Advanced buy-side diagnostics | Computes SMA 5/125 equity, underwater drawdown, rolling volatility and exposure state from live SPY prices. |
| [66_factor_exposure_diagnostics.ipynb](66_factor_exposure_diagnostics.ipynb) | [PNG](../plots/66_factor_exposure_diagnostics_output_01.png) | Advanced buy-side diagnostics | Estimates rolling 126-day factor betas and recent contribution proxy for a mega-cap equity basket. |

## Run

```bash
export QJ_API_KEY="qj_..."
jupyter lab _candidates
```

Candidate notebooks use optional API calls through `safe_call(...)` when a feed may depend on tenant entitlements.
