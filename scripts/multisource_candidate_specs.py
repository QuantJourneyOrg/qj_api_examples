"""Multi-source candidate notebook specs numbered 80+."""

from __future__ import annotations


def get_multisource_workflow_specs(NotebookSpec):
    """Return high-signal buy-side workflow candidates.

    The builder passes in its local NotebookSpec dataclass to avoid importing the
    builder module back from this spec-only file.
    """

    return [
        NotebookSpec(
            "80_investment_evidence_packet.ipynb",
            "Investment Evidence Packet",
            "Multi-source evidence workflow",
            "Assembles pricing, fundamentals, filings, insiders, identity, short-interest and volatility context for one investable name.",
            [
                "qj.eod.get_historical_prices",
                "qj.fmp.get_financial_ratios_ttm",
                "qj.sec.get_company_filings",
                "qj.sec.get_insider_transactions",
                "qj.openfigi.get_figi_data",
                "qj.finra.get_short_interest",
                "qj.cboe.get_vix_data",
            ],
            [
                """
                symbol = "AAPL"
                benchmark = "SPY"
                prices, volumes = price_panel([symbol, benchmark], start="2022-01-01", end=END)
                ratios_raw = safe_call("FMP TTM ratios", qj.fmp.get_financial_ratios_ttm, symbol=symbol)
                filings_raw = safe_call("SEC company filings", qj.sec.get_company_filings, symbol=symbol, limit=20)
                insiders_raw = safe_call("SEC insider transactions", qj.sec.get_insider_transactions, symbol=symbol, limit=100)
                figi_raw = safe_call("OpenFIGI identity", qj.openfigi.get_figi_data, symbol=symbol, exchange="US")
                short_raw = safe_call("FINRA short interest", qj.finra.get_short_interest, symbol=symbol)
                vix_raw = safe_call("CBOE VIX", qj.cboe.get_vix_data, start_date="2022-01-01", end_date=END)
                """,
                """
                def first_dict(payload: Any) -> dict[str, Any]:
                    value = unwrap(payload)
                    if isinstance(value, list) and value:
                        return value[0] if isinstance(value[0], dict) else {}
                    return value if isinstance(value, dict) else {}

                def pick_number(row: dict[str, Any], keys: list[str]) -> float:
                    for key in keys:
                        if key in row:
                            return pd.to_numeric(row.get(key), errors="coerce")
                    return np.nan

                ratios = first_dict(ratios_raw)
                figi = first_dict(figi_raw)
                filings = pd.DataFrame(as_rows(filings_raw))
                insiders = pd.DataFrame(as_rows(insiders_raw))
                short_interest = pd.DataFrame(as_rows(short_raw))
                ret = returns(prices)
                """,
                """
                evidence = pd.Series({
                    "latest_price": prices[symbol].dropna().iloc[-1],
                    "return_126d": prices[symbol].pct_change(126).iloc[-1],
                    "volatility_63d": ret[symbol].tail(63).std() * np.sqrt(252),
                    "beta_to_spy_252d": ret[[symbol, benchmark]].tail(252).cov().loc[symbol, benchmark] / ret[benchmark].tail(252).var(),
                    "pe_ttm": pick_number(ratios, ["peRatioTTM", "pe_ttm", "priceEarningsRatioTTM"]),
                    "gross_margin_ttm": pick_number(ratios, ["grossProfitMarginTTM", "gross_margin_ttm"]),
                    "recent_filings": len(filings),
                    "insider_events": len(insiders),
                    "short_interest_rows": len(short_interest),
                })
                identity = pd.Series({
                    "composite_figi": figi.get("composite_figi") or figi.get("compositeFIGI") or figi.get("figi"),
                    "share_class_figi": figi.get("share_class_figi") or figi.get("shareClassFIGI"),
                    "security_type": figi.get("security_type") or figi.get("securityType"),
                    "currency": figi.get("currency"),
                }).dropna()
                display(evidence)
                display(identity)
                prices[[symbol, benchmark]].div(prices[[symbol, benchmark]].iloc[0]).plot(title="Evidence packet price context")
                plt.ylabel("normalized value")
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "81_pre_trade_liquidity_capacity_check.ipynb",
            "Pre-Trade Liquidity and Capacity Check",
            "Pre-trade risk workflow",
            "Combines prices, dollar volume, short-interest context, sector metadata and valuation fields into a trade-capacity screen.",
            [
                "qj.eod.get_historical_prices",
                "qj.finra.get_short_interest",
                "qj.yf.get_sp500_sectors",
                "qj.fmp.get_stock_screener",
                "qj.fmp.get_financial_ratios_ttm",
            ],
            [
                """
                universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "JPM", "LLY", "XOM"]
                prices, volumes = price_panel(universe, start="2023-01-01", end=END)
                sectors_raw = safe_call("S&P 500 sectors", qj.yf.get_sp500_sectors)
                screener_raw = safe_call("FMP large-cap screener", qj.fmp.get_stock_screener, marketCapMoreThan=10_000_000_000, limit=100)
                short_raw = {symbol: safe_call(f"FINRA short interest {symbol}", qj.finra.get_short_interest, symbol=symbol) for symbol in universe[:6]}
                """,
                """
                ret = returns(prices)
                adv = dollar_adv(prices, volumes).iloc[-1]
                vol_63d = ret.tail(63).std() * np.sqrt(252)
                ratio_rows = []
                for symbol in universe:
                    row = unwrap(safe_call(f"FMP ratios {symbol}", qj.fmp.get_financial_ratios_ttm, symbol=symbol)) or {}
                    if isinstance(row, list):
                        row = row[0] if row else {}
                    ratio_rows.append({
                        "symbol": symbol,
                        "pe_ttm": pd.to_numeric(row.get("peRatioTTM"), errors="coerce"),
                        "gross_margin_ttm": pd.to_numeric(row.get("grossProfitMarginTTM"), errors="coerce"),
                    })
                ratios = pd.DataFrame(ratio_rows).set_index("symbol")
                """,
                """
                order_size = 25_000_000
                capacity = pd.DataFrame({
                    "adv_usd": adv,
                    "capacity_5pct_adv": adv * 0.05,
                    "capacity_10pct_adv": adv * 0.10,
                    "volatility_63d": vol_63d,
                    "momentum_126d": prices.pct_change(126).iloc[-1],
                }).join(ratios)
                capacity["days_to_trade_10pct_adv"] = order_size / capacity["capacity_10pct_adv"]
                capacity["liquidity_flag"] = np.where(capacity["days_to_trade_10pct_adv"] > 3, "stagger", "ok")
                capacity["short_feed_rows"] = pd.Series({symbol: len(as_rows(payload)) for symbol, payload in short_raw.items()})
                display(capacity.sort_values("days_to_trade_10pct_adv", ascending=False))
                capacity["days_to_trade_10pct_adv"].sort_values(ascending=False).plot(kind="bar", title="Days to trade at 10% ADV")
                plt.ylabel("days")
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "82_earnings_event_pead_research.ipynb",
            "Earnings Event PEAD Research",
            "Event research workflow",
            "Joins earnings calendar, earnings surprises, fundamentals and prices to study post-earnings announcement drift across a peer set.",
            [
                "qj.fmp.get_earnings_calendar",
                "qj.fmp.get_earnings_surprises",
                "qj.fmp.get_financial_ratios_ttm",
                "qj.eod.get_historical_prices",
            ],
            [
                """
                symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META"]
                calendar = safe_call("FMP earnings calendar", qj.fmp.get_earnings_calendar, from_date="2023-01-01", to_date=END)
                surprises = {symbol: safe_call(f"FMP earnings surprises {symbol}", qj.fmp.get_earnings_surprises, symbol=symbol) for symbol in symbols}
                ratios = {symbol: safe_call(f"FMP ratios {symbol}", qj.fmp.get_financial_ratios_ttm, symbol=symbol) for symbol in symbols}
                prices, volumes = price_panel(symbols, start="2022-01-01", end=END)
                """,
                """
                event_rows = []
                for symbol, payload in surprises.items():
                    for item in as_rows(payload):
                        event_date = pd.to_datetime(item.get("date") or item.get("fiscalDateEnding"), errors="coerce")
                        surprise = pd.to_numeric(item.get("surprisePercentage") or item.get("surprise"), errors="coerce")
                        if pd.notna(event_date):
                            event_rows.append({"symbol": symbol, "event_date": event_date, "surprise": surprise})
                events = pd.DataFrame(event_rows)
                if events.empty:
                    events = pd.DataFrame({"symbol": symbols, "event_date": [prices.index[-90]] * len(symbols), "surprise": np.nan})
                """,
                """
                curves = []
                event_metrics = []
                for row in events.dropna(subset=["event_date"]).itertuples():
                    if row.symbol not in prices:
                        continue
                    idx = prices.index.searchsorted(row.event_date)
                    if idx < 10 or idx + 42 >= len(prices):
                        continue
                    window = prices[row.symbol].iloc[idx - 10:idx + 43]
                    curve = window / window.iloc[10] - 1
                    curves.append(pd.Series(curve.values, index=range(-10, len(curve) - 10), name=row.symbol))
                    event_metrics.append({"symbol": row.symbol, "event_date": row.event_date, "surprise": row.surprise, "fwd_5d": curve.iloc[15], "fwd_21d": curve.iloc[31], "fwd_42d": curve.iloc[-1]})
                event_curve = pd.concat(curves, axis=1) if curves else pd.DataFrame()
                metrics = pd.DataFrame(event_metrics)
                display(metrics.sort_values("fwd_21d", ascending=False).head(20))
                if not event_curve.empty:
                    event_curve.mean(axis=1).plot(title="Average PEAD curve")
                    plt.axvline(0, color="black", linestyle="--", alpha=0.5)
                    plt.ylabel("return vs event day")
                    plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "83_portfolio_morning_risk_brief.ipynb",
            "Portfolio Morning Risk Brief",
            "PM workflow",
            "Builds a daily PM packet from holdings, price moves, factor proxies, macro rates, VIX, filings, insiders and earnings context.",
            [
                "qj.eod.get_historical_prices",
                "qj.cboe.get_vix_data",
                "qj.fred.get_treasury_2y",
                "qj.fred.get_treasury_10y",
                "qj.sec.get_recent_filings",
                "qj.insider.latest_trades",
                "qj.fmp.get_earnings_calendar",
                "qj.ff.get_factors",
            ],
            [
                """
                holdings = pd.Series({"AAPL": 0.18, "MSFT": 0.17, "NVDA": 0.15, "GOOGL": 0.12, "AMZN": 0.12, "META": 0.10, "JPM": 0.08, "XOM": 0.08})
                factors = ["SPY", "QQQ", "IWM", "TLT", "GLD", "UUP"]
                prices, volumes = price_panel(list(holdings.index) + factors, start="2022-01-01", end=END)
                vix = safe_call("CBOE VIX", qj.cboe.get_vix_data, start_date="2022-01-01", end_date=END)
                rates = {
                    "2y": safe_call("FRED 2Y", qj.fred.get_treasury_2y),
                    "10y": safe_call("FRED 10Y", qj.fred.get_treasury_10y),
                }
                filings = safe_call("SEC recent filings", qj.sec.get_recent_filings, limit=30)
                insiders = safe_call("Insider latest trades", qj.insider.latest_trades, symbols=list(holdings.index), limit=50)
                earnings = safe_call("FMP earnings calendar", qj.fmp.get_earnings_calendar, from_date=END, to_date=END)
                ff = safe_call("Fama-French factors", qj.ff.get_factors, region="US")
                """,
                """
                ret = returns(prices)
                book_ret = portfolio_returns(ret, holdings)
                latest_returns = ret[list(holdings.index)].iloc[-1]
                contribution = holdings * latest_returns
                rc = risk_contribution(ret[list(holdings.index)], holdings)
                betas = rolling_betas(book_ret, ret[factors], window=126)
                brief = pd.DataFrame({
                    "weight": holdings,
                    "latest_return": latest_returns,
                    "daily_contribution": contribution,
                    "risk_contribution": rc,
                    "momentum_63d": prices[list(holdings.index)].pct_change(63).iloc[-1],
                }).sort_values("daily_contribution")
                """,
                """
                packet = {
                    "portfolio_return_latest": float(book_ret.iloc[-1]),
                    "volatility_63d": float(book_ret.tail(63).std() * np.sqrt(252)),
                    "max_drawdown_1y": max_drawdown((1 + book_ret.tail(252)).cumprod()),
                    "filing_rows": len(as_rows(filings)),
                    "insider_rows": len(as_rows(insiders)),
                    "earnings_rows": len(as_rows(earnings)),
                }
                display(pd.Series(packet))
                display(brief)
                if not betas.empty:
                    display(betas.tail(1).T.rename(columns={betas.index[-1]: "latest_beta"}))
                brief[["daily_contribution", "risk_contribution"]].plot(kind="bar", title="Morning contribution and risk")
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "84_crowding_smart_money_flow.ipynb",
            "Crowding and Smart-Money Flow",
            "Ownership and flow workflow",
            "Combines institutional holders, SEC ownership, congressional trades, insiders, short interest and returns to flag crowded or consensus trades.",
            [
                "qj.fmp.get_institutional_holders",
                "qj.sec.get_institutional_holdings",
                "qj.fmp.get_senate_trades",
                "qj.fmp.get_house_trades",
                "qj.sec.get_insider_transactions",
                "qj.finra.get_short_interest",
                "qj.eod.get_historical_prices",
            ],
            [
                """
                symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
                holders = {symbol: safe_call(f"FMP institutional holders {symbol}", qj.fmp.get_institutional_holders, symbol=symbol) for symbol in symbols}
                sec_holdings = {symbol: safe_call(f"SEC institutional holdings {symbol}", qj.sec.get_institutional_holdings, symbol=symbol) for symbol in symbols[:3]}
                congress = {
                    "senate": safe_call("FMP senate trades", qj.fmp.get_senate_trades, symbol="AAPL"),
                    "house": safe_call("FMP house trades", qj.fmp.get_house_trades, symbol="AAPL"),
                }
                insiders = {symbol: safe_call(f"SEC insiders {symbol}", qj.sec.get_insider_transactions, symbol=symbol, limit=50) for symbol in symbols[:4]}
                short_interest = {symbol: safe_call(f"FINRA short interest {symbol}", qj.finra.get_short_interest, symbol=symbol) for symbol in symbols[:4]}
                prices, volumes = price_panel(symbols, start="2023-01-01", end=END)
                """,
                """
                rows = []
                for symbol in symbols:
                    holder_rows = as_rows(holders.get(symbol))
                    values = []
                    for item in holder_rows:
                        values.append(pd.to_numeric(item.get("value") or item.get("marketValue") or item.get("shares"), errors="coerce"))
                    values = pd.Series(values).dropna()
                    rows.append({
                        "symbol": symbol,
                        "holder_count": len(holder_rows),
                        "top10_concentration": values.nlargest(10).sum() / values.sum() if values.sum() else np.nan,
                        "sec_holding_rows": len(as_rows(sec_holdings.get(symbol))),
                        "insider_rows": len(as_rows(insiders.get(symbol))),
                        "short_rows": len(as_rows(short_interest.get(symbol))),
                    })
                crowding = pd.DataFrame(rows).set_index("symbol")
                crowding["momentum_126d"] = prices.pct_change(126).iloc[-1].reindex(crowding.index)
                crowding["volatility_63d"] = returns(prices).tail(63).std().reindex(crowding.index) * np.sqrt(252)
                crowding["crowding_score"] = crowding["holder_count"].rank(pct=True) + crowding["top10_concentration"].rank(pct=True) + crowding["momentum_126d"].rank(pct=True)
                display(crowding.sort_values("crowding_score", ascending=False))
                """,
                """
                congress_rows = pd.concat([pd.DataFrame(as_rows(payload)).assign(source=source) for source, payload in congress.items()], ignore_index=True)
                display(congress_rows.head())
                crowding[["holder_count", "top10_concentration", "momentum_126d", "crowding_score"]].plot(kind="bar", subplots=True, layout=(1, 4), figsize=(16, 4), title="Crowding and smart-money diagnostics")
                plt.tight_layout()
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "85_macro_shock_cross_asset_scenario.ipynb",
            "Macro Shock Cross-Asset Scenario",
            "Macro risk workflow",
            "Maps rate, dollar, volatility and commodity shocks onto a cross-asset portfolio using observed factor relationships.",
            [
                "qj.eod.get_historical_prices",
                "qj.fred.get_cpi",
                "qj.fred.get_treasury_2y",
                "qj.fred.get_treasury_10y",
                "qj.cboe.get_vix_data",
                "qj.cftc.get_cot_summary",
            ],
            [
                """
                portfolio = pd.Series({"SPY": 0.42, "TLT": 0.22, "GLD": 0.14, "DBC": 0.12, "UUP": 0.10})
                factors = ["SPY", "TLT", "GLD", "DBC", "UUP"]
                prices, volumes = price_panel(factors, start="2018-01-01", end=END)
                macro = {
                    "cpi": safe_call("FRED CPI", qj.fred.get_cpi),
                    "2y": safe_call("FRED 2Y", qj.fred.get_treasury_2y),
                    "10y": safe_call("FRED 10Y", qj.fred.get_treasury_10y),
                    "vix": safe_call("CBOE VIX", qj.cboe.get_vix_data, start_date="2018-01-01", end_date=END),
                    "cot_spx": safe_call("CFTC COT SPX", qj.cftc.get_cot_summary, symbol="SPX"),
                    "cot_gold": safe_call("CFTC COT gold", qj.cftc.get_cot_summary, symbol="GC"),
                }
                """,
                """
                ret = returns(prices).dropna()
                base_ret = portfolio_returns(ret, portfolio)
                shocks = pd.DataFrame({
                    "inflation_spike": {"SPY": -0.045, "TLT": -0.070, "GLD": 0.030, "DBC": 0.085, "UUP": 0.020},
                    "hard_landing": {"SPY": -0.100, "TLT": 0.055, "GLD": 0.040, "DBC": -0.060, "UUP": 0.030},
                    "dollar_squeeze": {"SPY": -0.055, "TLT": -0.015, "GLD": -0.025, "DBC": -0.035, "UUP": 0.060},
                    "risk_on_easing": {"SPY": 0.075, "TLT": 0.020, "GLD": -0.015, "DBC": 0.035, "UUP": -0.025},
                }).T
                scenario_return = shocks @ portfolio
                marginal = shocks.mul(portfolio, axis=1)
                display((scenario_return * 100).rename("scenario_return_pct"))
                display(marginal.mul(100))
                """,
                """
                scenario_return.mul(100).plot(kind="bar", title="Cross-asset macro shock scenario")
                plt.ylabel("portfolio return proxy (%)")
                plt.show()
                plot_nav({"base portfolio": base_ret, "SPY": ret["SPY"]}, "Base portfolio context")
                """,
            ],
        ),
        NotebookSpec(
            "86_macro_regime_allocation_control.ipynb",
            "Macro Regime Allocation Control",
            "Macro allocation workflow",
            "Classifies growth, inflation, rates and volatility regimes, then translates the state into allocation tilts.",
            [
                "qj.fred.get_cpi",
                "qj.fred.get_fred_data_series_by_id",
                "qj.fred.get_treasury_2y",
                "qj.fred.get_treasury_10y",
                "qj.cboe.get_vix_data",
                "qj.ff.get_factors",
                "qj.eod.get_historical_prices",
            ],
            [
                """
                assets = ["SPY", "TLT", "GLD", "DBC", "UUP"]
                macro_raw = {
                    "cpi": safe_call("FRED CPI", qj.fred.get_cpi),
                    "unemployment": safe_call("FRED unemployment", qj.fred.get_fred_data_series_by_id, series_id="UNRATE"),
                    "industrial_production": safe_call("FRED industrial production", qj.fred.get_fred_data_series_by_id, series_id="INDPRO"),
                    "2y": safe_call("FRED 2Y", qj.fred.get_treasury_2y),
                    "10y": safe_call("FRED 10Y", qj.fred.get_treasury_10y),
                    "vix": safe_call("CBOE VIX", qj.cboe.get_vix_data, start_date="2018-01-01", end_date=END),
                    "ff": safe_call("Fama-French factors", qj.ff.get_factors, region="US"),
                }
                prices, volumes = price_panel(assets, start="2018-01-01", end=END)
                """,
                """
                ret = returns(prices).dropna()
                trend = prices.pct_change(126).mean(axis=1)
                inflation_proxy = ret["DBC"].rolling(63).sum() - ret["TLT"].rolling(63).sum()
                vol_proxy = ret["SPY"].rolling(21).std() * np.sqrt(252)
                rates_proxy = ret["TLT"].rolling(63).sum() * -1
                regime = pd.DataFrame({
                    "growth": np.where(trend > 0, "growth_up", "growth_down"),
                    "inflation": np.where(inflation_proxy > 0, "inflation_up", "inflation_down"),
                    "rates": np.where(rates_proxy > 0, "rates_up", "rates_down"),
                    "vol": np.where(vol_proxy > vol_proxy.rolling(252).median(), "vol_high", "vol_normal"),
                }, index=ret.index)
                """,
                """
                weights = pd.DataFrame(index=ret.index, columns=assets, dtype=float)
                weights.loc[:, :] = [0.45, 0.20, 0.12, 0.12, 0.11]
                risk_off = (regime["growth"] == "growth_down") | (regime["vol"] == "vol_high")
                inflation_up = regime["inflation"] == "inflation_up"
                weights.loc[risk_off, ["SPY"]] -= 0.18
                weights.loc[risk_off, ["TLT", "GLD", "UUP"]] += [0.08, 0.06, 0.04]
                weights.loc[inflation_up, ["TLT"]] -= 0.08
                weights.loc[inflation_up, ["GLD", "DBC"]] += [0.04, 0.04]
                weights = weights.clip(lower=0.02).div(weights.sum(axis=1), axis=0)
                regime_ret = (weights.shift(1) * ret).sum(axis=1)
                equal_ret = ret.mean(axis=1)
                display(regime.tail())
                display(weights.tail())
                plot_nav({"macro regime allocation": regime_ret, "equal-weight": equal_ret}, "Macro regime allocation")
                weights.tail(504).plot(title="Recent allocation tilts")
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "87_inflation_shock_monitor.ipynb",
            "Inflation Shock Monitor",
            "Macro monitoring workflow",
            "Joins inflation, rates, energy, commodities and sector proxies to estimate where a portfolio is exposed to inflation surprises.",
            [
                "qj.fred.get_cpi",
                "qj.fred.get_fred_data_series_by_id",
                "qj.fred.get_effective_federal_funds_rate",
                "qj.fred.get_treasury_2y",
                "qj.fred.get_treasury_10y",
                "qj.eia.get_petroleum_prices",
                "qj.eod.get_historical_prices",
            ],
            [
                """
                sector_etfs = ["XLE", "XLK", "XLF", "XLU", "XLP", "XLY"]
                macro_raw = {
                    "cpi": safe_call("FRED CPI", qj.fred.get_cpi),
                    "pce": safe_call("FRED PCE price index", qj.fred.get_fred_data_series_by_id, series_id="PCEPI"),
                    "fed_funds": safe_call("FRED fed funds", qj.fred.get_effective_federal_funds_rate),
                    "2y": safe_call("FRED 2Y", qj.fred.get_treasury_2y),
                    "10y": safe_call("FRED 10Y", qj.fred.get_treasury_10y),
                    "oil": safe_call("EIA petroleum prices", qj.eia.get_petroleum_prices),
                }
                prices, volumes = price_panel(sector_etfs + ["DBC", "UUP", "TLT"], start="2018-01-01", end=END)
                """,
                """
                ret = returns(prices).dropna()
                inflation_proxy = (ret["DBC"].rolling(63).sum() - ret["TLT"].rolling(63).sum()).dropna()
                sector_betas = {}
                for symbol in sector_etfs:
                    aligned = pd.concat([ret[symbol], inflation_proxy], axis=1).dropna()
                    if aligned.empty:
                        sector_betas[symbol] = np.nan
                    else:
                        sector_betas[symbol] = aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var()
                beta_table = pd.Series(sector_betas).rename("inflation_beta_proxy").sort_values(ascending=False)
                """,
                """
                monitor = pd.DataFrame({
                    "sector_return_63d": prices[sector_etfs].pct_change(63).iloc[-1],
                    "volatility_63d": ret[sector_etfs].tail(63).std() * np.sqrt(252),
                    "inflation_beta_proxy": beta_table,
                }).sort_values("inflation_beta_proxy", ascending=False)
                display(monitor)
                monitor["inflation_beta_proxy"].plot(kind="bar", title="Sector inflation beta proxy")
                plt.ylabel("beta to commodity-duration proxy")
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "88_macro_positioning_cot_dashboard.ipynb",
            "Macro Positioning COT Dashboard",
            "Macro positioning workflow",
            "Uses CFTC positioning, cross-asset prices, rates and volatility to identify crowded macro exposures.",
            [
                "qj.cftc.get_cot_summary",
                "qj.eod.get_historical_prices",
                "qj.fred.get_treasury_10y",
                "qj.cboe.get_vix_data",
            ],
            [
                """
                markets = {"SPX": "equity_index", "GC": "gold", "CL": "crude_oil", "DX": "usd", "ZN": "10y_note"}
                cot = {symbol: safe_call(f"CFTC COT {symbol}", qj.cftc.get_cot_summary, symbol=symbol) for symbol in markets}
                assets = ["SPY", "GLD", "USO", "UUP", "TLT", "DBC"]
                prices, volumes = price_panel(assets, start="2018-01-01", end=END)
                ten_y = safe_call("FRED 10Y", qj.fred.get_treasury_10y)
                vix = safe_call("CBOE VIX", qj.cboe.get_vix_data, start_date="2018-01-01", end_date=END)
                """,
                """
                ret = returns(prices).dropna()
                rows = []
                for symbol, payload in cot.items():
                    data = unwrap(payload)
                    row = data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else {}
                    numeric = {k: pd.to_numeric(v, errors="coerce") for k, v in row.items() if isinstance(v, (int, float, str))}
                    long_val = next((v for k, v in numeric.items() if "long" in k.lower() and pd.notna(v)), np.nan)
                    short_val = next((v for k, v in numeric.items() if "short" in k.lower() and pd.notna(v)), np.nan)
                    net = long_val - short_val if pd.notna(long_val) and pd.notna(short_val) else np.nan
                    rows.append({"cot_symbol": symbol, "market": markets[symbol], "long": long_val, "short": short_val, "net_position": net})
                positioning = pd.DataFrame(rows).set_index("cot_symbol")
                """,
                """
                cross_asset = pd.DataFrame({
                    "return_63d": prices.pct_change(63).iloc[-1],
                    "volatility_63d": ret.tail(63).std() * np.sqrt(252),
                    "momentum_z": zscore(prices.pct_change(126).mean(axis=1), 252).iloc[-1],
                })
                display(positioning)
                display(cross_asset)
                cross_asset[["return_63d", "volatility_63d"]].plot(kind="bar", title="Macro market context around COT positioning")
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "89_usd_liquidity_cross_asset_mosaic.ipynb",
            "USD Liquidity Cross-Asset Mosaic",
            "Multi-source macro liquidity workflow",
            "Combines Fed liquidity series, rates, volatility, equity, duration, gold, dollar and crypto prices into one data mosaic.",
            [
                "qj.fred.get_fred_data_series_by_id",
                "qj.fred.get_effective_federal_funds_rate",
                "qj.fred.get_treasury_10y",
                "qj.fred.get_treasury_repo_rate",
                "qj.cboe.get_vix_data",
                "qj.eod.get_historical_prices",
                "qj.ccxt.get_historical_prices",
            ],
            [
                """
                macro_calls = {
                    "Fed balance sheet assets (FRED:WALCL)": qj.fred.get_fred_data_series_by_id(series_id="WALCL", start="2020-01-01"),
                    "Treasury General Account (FRED:WTREGEN)": qj.fred.get_fred_data_series_by_id(series_id="WTREGEN", start="2020-01-01"),
                    "Reverse repo operations (FRED:RRPONTSYD)": qj.fred.get_fred_data_series_by_id(series_id="RRPONTSYD", start="2020-01-01"),
                    "Effective Fed Funds Rate (FRED:FEDFUNDS)": qj.fred.get_effective_federal_funds_rate(start_date="2020-01-01"),
                    "10-Year Treasury Yield (FRED:DGS10)": qj.fred.get_treasury_10y(start_date="2020-01-01"),
                    "Treasury repo rate": qj.fred.get_treasury_repo_rate(start_date="2020-01-01"),
                }
                vix_raw = qj.cboe.get_vix_data(start_date="2020-01-01", end_date=END)
                prices, volumes = price_panel(["SPY", "TLT", "GLD", "UUP"], start="2020-01-01", end=END)
                btc_raw = qj.ccxt.get_historical_prices(symbol="BTC/USDT", exchange="binance", timeframe="1d", since="2020-01-01")
                """,
                """
                def series_from_payload(name: str, payload: Any) -> pd.Series:
                    frame = pd.DataFrame(as_rows(payload))
                    if frame.empty:
                        return pd.Series(dtype=float, name=name)
                    date_col = next((col for col in frame.columns if "date" in str(col).lower() or str(col).lower() in {"observation_date", "time"}), frame.columns[0])
                    numeric_cols = frame.select_dtypes(include="number").columns.tolist()
                    value_col = "value" if "value" in frame.columns else ("close" if "close" in frame.columns else (numeric_cols[-1] if numeric_cols else None))
                    if value_col is None:
                        return pd.Series(dtype=float, name=name)
                    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce")
                    frame[name] = pd.to_numeric(frame[value_col], errors="coerce")
                    return frame.dropna(subset=["date", name]).set_index("date")[name].sort_index()

                macro = pd.concat([series_from_payload(name, payload) for name, payload in macro_calls.items()], axis=1).dropna(how="all")
                vix = series_from_payload("CBOE Volatility Index (VIX)", vix_raw)
                btc = series_from_payload("Bitcoin spot (CCXT:BTC/USDT)", btc_raw)
                asset_prices = prices.join(btc, how="outer").dropna(how="all")
                """,
                """
                if macro.empty or asset_prices.empty:
                    raise RuntimeError("Liquidity mosaic requires macro observations and market prices")
                liquidity_index = (
                    macro.filter(like="Fed balance").ffill().iloc[:, 0].pct_change(26)
                    - macro.filter(like="Treasury General").ffill().iloc[:, 0].pct_change(26).reindex(macro.index)
                    - macro.filter(like="Reverse repo").ffill().iloc[:, 0].pct_change(26).reindex(macro.index)
                ).rename("USD liquidity impulse")
                cross_asset = asset_prices.pct_change(63).iloc[-1].rename("63d_return")
                risk_state = pd.concat([liquidity_index, vix.reindex(liquidity_index.index).ffill()], axis=1).dropna()
                display(macro.tail())
                display(cross_asset.sort_values(ascending=False))
                risk_state.tail(260).plot(title="USD liquidity impulse and VIX")
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "90_earnings_options_vol_reaction.ipynb",
            "Earnings Options Volatility Reaction",
            "Multi-source event and derivatives workflow",
            "Combines earnings calendar, surprises, option chain context, volatility feeds and post-event price reaction across a peer set.",
            [
                "qj.fmp.get_earnings_calendar",
                "qj.fmp.get_earnings_surprises",
                "qj.fmp.get_financial_ratios_ttm",
                "qj.cboe.get_options_expirations",
                "qj.cboe.get_options_chain",
                "qj.cboe.get_vix_data",
                "qj.eod.get_historical_prices",
            ],
            [
                """
                symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
                calendar_raw = qj.fmp.get_earnings_calendar(from_date="2024-01-01", to_date=END)
                surprise_raw = {symbol: qj.fmp.get_earnings_surprises(symbol=symbol) for symbol in symbols}
                ratio_raw = {symbol: qj.fmp.get_financial_ratios_ttm(symbol=symbol) for symbol in symbols}
                expirations_raw = qj.cboe.get_options_expirations(symbol="SPY")
                chain_raw = qj.cboe.get_options_chain(symbol="SPY")
                vix_raw = qj.cboe.get_vix_data(start_date="2024-01-01", end_date=END)
                prices, volumes = price_panel(symbols, start="2023-01-01", end=END)
                """,
                """
                events = []
                for symbol, payload in surprise_raw.items():
                    for row in as_rows(payload):
                        date = pd.to_datetime(row.get("date") or row.get("fiscalDateEnding") or row.get("reportedDate"), errors="coerce")
                        surprise = pd.to_numeric(row.get("surprisePercentage") or row.get("surprise") or row.get("epsSurprise"), errors="coerce")
                        if pd.notna(date):
                            events.append({"symbol": symbol, "event_date": date, "surprise": surprise})
                events = pd.DataFrame(events)
                if events.empty:
                    raise RuntimeError("No earnings surprise rows returned")
                """,
                """
                curves = []
                rows = []
                for row in events.itertuples():
                    if row.symbol not in prices:
                        continue
                    position = prices.index.searchsorted(row.event_date)
                    if position < 10 or position + 22 >= len(prices):
                        continue
                    window = prices[row.symbol].iloc[position - 10:position + 22]
                    event_curve = window / window.iloc[10] - 1
                    curves.append(pd.Series(event_curve.values, index=range(-10, len(event_curve) - 10), name=row.symbol))
                    rows.append({"symbol": row.symbol, "surprise_pct": row.surprise, "realized_1d": event_curve.iloc[11], "realized_5d": event_curve.iloc[15], "realized_21d": event_curve.iloc[-1]})
                reaction = pd.DataFrame(rows)
                event_curve = pd.concat(curves, axis=1) if curves else pd.DataFrame()
                chain = pd.DataFrame(as_rows(chain_raw))
                expirations = pd.DataFrame(as_rows(expirations_raw))
                vix = pd.DataFrame(as_rows(vix_raw))
                display(pd.Series({"calendar_rows": len(as_rows(calendar_raw)), "surprise_events": len(events), "option_chain_rows": len(chain), "expiration_rows": len(expirations), "vix_rows": len(vix)}))
                display(reaction.sort_values("realized_21d", ascending=False))
                if not event_curve.empty:
                    event_curve.mean(axis=1).plot(title="Average earnings reaction curve")
                    plt.axvline(0, color="black", linestyle="--", alpha=0.5)
                    plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "91_sec_filing_fundamental_price_mosaic.ipynb",
            "SEC Filing Fundamental Price Mosaic",
            "Multi-source disclosure and fundamentals workflow",
            "Joins SEC filings, company facts, FMP statements, ratios, analyst estimates, identity and price-volume reaction for one issuer.",
            [
                "qj.sec.get_company_filings",
                "qj.sec.get_company_facts",
                "qj.sec.get_company_submissions",
                "qj.fmp.get_income_statement",
                "qj.fmp.get_balance_sheet_statement",
                "qj.fmp.get_financial_ratios_ttm",
                "qj.fmp.get_analyst_estimates",
                "qj.openfigi.get_figi_data",
                "qj.eod.get_historical_prices",
            ],
            [
                """
                symbol = "AAPL"
                filings_raw = qj.sec.get_company_filings(symbol=symbol, limit=40)
                facts_raw = qj.sec.get_company_facts(symbol=symbol)
                submissions_raw = qj.sec.get_company_submissions(symbol=symbol)
                income_raw = qj.fmp.get_income_statement(symbol=symbol, period="annual", limit=5)
                balance_raw = qj.fmp.get_balance_sheet_statement(symbol=symbol, period="annual", limit=5)
                ratios_raw = qj.fmp.get_financial_ratios_ttm(symbol=symbol)
                estimates_raw = qj.fmp.get_analyst_estimates(symbol=symbol, period="annual", limit=8)
                identity_raw = qj.openfigi.get_figi_data(symbol=symbol, exchange="US")
                prices = price_frame(symbol, start="2021-01-01", end=END)
                """,
                """
                filings = pd.DataFrame(as_rows(filings_raw))
                income = pd.DataFrame(as_rows(income_raw))
                balance = pd.DataFrame(as_rows(balance_raw))
                estimates = pd.DataFrame(as_rows(estimates_raw))
                ratios = pd.DataFrame(as_rows(ratios_raw))
                identity = pd.DataFrame(as_rows(identity_raw))
                if filings.empty:
                    raise RuntimeError("No SEC filing rows returned")
                """,
                """
                event_dates = pd.to_datetime(
                    filings.get("filingDate", filings.get("filedAt", filings.get("date"))),
                    errors="coerce",
                ).dropna()
                reactions = []
                for event_date in event_dates.head(20):
                    pos = prices.index.searchsorted(event_date)
                    if pos > 5 and pos + 5 < len(prices):
                        pre = prices["price"].iloc[pos - 5]
                        post = prices["price"].iloc[pos + 5]
                        reactions.append({"filing_date": event_date, "five_day_reaction": post / pre - 1})
                reaction = pd.DataFrame(reactions)
                summary = pd.Series({
                    "sec_filings": len(filings),
                    "company_fact_rows": len(as_rows(facts_raw)),
                    "income_statement_rows": len(income),
                    "balance_sheet_rows": len(balance),
                    "estimate_rows": len(estimates),
                    "identity_rows": len(identity),
                    "median_5d_filing_reaction": reaction["five_day_reaction"].median() if not reaction.empty else np.nan,
                })
                display(summary)
                display(filings.head())
                prices["price"].tail(756).plot(title=f"{symbol} price with SEC filing events")
                for event_date in event_dates.head(12):
                    plt.axvline(event_date, color="tab:pink", alpha=0.35)
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "92_macro_rates_equity_factor_panel.ipynb",
            "Macro Rates Equity Factor Panel",
            "Multi-source macro and factor workflow",
            "Links inflation, labor, rates, breakevens, Fama-French factors and equity sector returns into a regime-conditioned factor panel.",
            [
                "qj.fred.get_cpi",
                "qj.fred.get_fred_data_series_by_id",
                "qj.fred.get_effective_federal_funds_rate",
                "qj.fred.get_treasury_2y",
                "qj.fred.get_treasury_10y",
                "qj.ff.get_factors",
                "qj.eod.get_historical_prices",
            ],
            [
                """
                def series_from_payload(name: str, payload: Any) -> pd.Series:
                    frame = pd.DataFrame(as_rows(payload))
                    if frame.empty:
                        return pd.Series(dtype=float, name=name)
                    date_col = next((col for col in frame.columns if "date" in str(col).lower() or str(col).lower() in {"observation_date", "time"}), frame.columns[0])
                    numeric_cols = frame.select_dtypes(include="number").columns.tolist()
                    value_col = "value" if "value" in frame.columns else ("close" if "close" in frame.columns else (numeric_cols[-1] if numeric_cols else None))
                    if value_col is None:
                        return pd.Series(dtype=float, name=name)
                    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce")
                    frame[name] = pd.to_numeric(frame[value_col], errors="coerce")
                    return frame.dropna(subset=["date", name]).set_index("date")[name].sort_index()
                """,
                """
                sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "XLI"]
                macro_raw = {
                    "CPI Urban Consumers YoY (FRED:CPIAUCSL)": qj.fred.get_cpi(start_date="2018-01-01"),
                    "Civilian Unemployment Rate (FRED:UNRATE)": qj.fred.get_fred_data_series_by_id(series_id="UNRATE", start="2018-01-01"),
                    "10Y Breakeven Inflation (FRED:T10YIE)": qj.fred.get_fred_data_series_by_id(series_id="T10YIE", start="2018-01-01"),
                    "Effective Fed Funds Rate (FRED:FEDFUNDS)": qj.fred.get_effective_federal_funds_rate(start_date="2018-01-01"),
                    "2-Year Treasury Yield (FRED:DGS2)": qj.fred.get_treasury_2y(start_date="2018-01-01"),
                    "10-Year Treasury Yield (FRED:DGS10)": qj.fred.get_treasury_10y(start_date="2018-01-01"),
                }
                ff_raw = qj.ff.get_factors(region="US")
                prices, volumes = price_panel(sector_etfs + ["SPY", "TLT"], start="2018-01-01", end=END)
                """,
                """
                macro = pd.concat([series_from_payload(name, payload) for name, payload in macro_raw.items()], axis=1).dropna(how="all")
                ret = returns(prices).dropna()
                factor_rows = pd.DataFrame(as_rows(ff_raw))
                if not factor_rows.empty:
                    date_col = next((col for col in factor_rows.columns if "date" in str(col).lower()), factor_rows.columns[0])
                    factor_rows["date"] = pd.to_datetime(factor_rows[date_col], errors="coerce")
                    factor_rows = factor_rows.dropna(subset=["date"]).set_index("date").sort_index()
                """,
                """
                rates_proxy = ret["TLT"].rolling(63).sum() * -1
                inflation_proxy = ret["XLE"].rolling(63).sum() - ret["TLT"].rolling(63).sum()
                regime = pd.DataFrame({
                    "rising_rate_regime": rates_proxy > rates_proxy.rolling(252).median(),
                    "inflation_pressure_regime": inflation_proxy > inflation_proxy.rolling(252).median(),
                }, index=ret.index).dropna()
                sector_returns = ret[sector_etfs].join(regime).dropna()
                conditioned = sector_returns.groupby(["rising_rate_regime", "inflation_pressure_regime"])[sector_etfs].mean() * 252
                display(macro.tail())
                display(factor_rows.tail() if not factor_rows.empty else pd.DataFrame())
                display(conditioned)
                conditioned.T.plot(kind="bar", title="Sector returns by rates and inflation regime")
                plt.ylabel("annualized return")
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "93_crypto_macro_liquidity_panel.ipynb",
            "Crypto Macro Liquidity Panel",
            "Multi-source crypto and macro workflow",
            "Combines exchange spot, funding, open interest, CBOE volatility, rates and dollar proxies for digital-asset exposure monitoring.",
            [
                "qj.ccxt.get_historical_prices",
                "qj.ccxt.get_historical_funding_rates",
                "qj.ccxt.get_open_interest",
                "qj.coingecko.get_historical_prices",
                "qj.cboe.get_vix_data",
                "qj.fred.get_treasury_10y",
                "qj.fred.get_effective_federal_funds_rate",
                "qj.eod.get_historical_prices",
            ],
            [
                """
                def series_from_payload(name: str, payload: Any) -> pd.Series:
                    frame = pd.DataFrame(as_rows(payload))
                    if frame.empty:
                        return pd.Series(dtype=float, name=name)
                    date_col = next((col for col in frame.columns if "date" in str(col).lower() or str(col).lower() in {"observation_date", "time"}), frame.columns[0])
                    numeric_cols = frame.select_dtypes(include="number").columns.tolist()
                    value_col = "value" if "value" in frame.columns else ("close" if "close" in frame.columns else (numeric_cols[-1] if numeric_cols else None))
                    if value_col is None:
                        return pd.Series(dtype=float, name=name)
                    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce")
                    frame[name] = pd.to_numeric(frame[value_col], errors="coerce")
                    return frame.dropna(subset=["date", name]).set_index("date")[name].sort_index()
                """,
                """
                btc_raw = qj.ccxt.get_historical_prices(symbol="BTC/USDT", exchange="binance", timeframe="1d", since="2020-01-01")
                eth_raw = qj.ccxt.get_historical_prices(symbol="ETH/USDT", exchange="binance", timeframe="1d", since="2020-01-01")
                funding_raw = qj.ccxt.get_historical_funding_rates(symbol="BTC/USDT", exchange="binance")
                open_interest_raw = qj.ccxt.get_open_interest(symbol="BTC/USDT", exchange="binance")
                gecko_raw = qj.coingecko.get_historical_prices(coin_id="bitcoin", vs_currency="usd", days="max")
                vix_raw = qj.cboe.get_vix_data(start_date="2020-01-01", end_date=END)
                ten_y_raw = qj.fred.get_treasury_10y(start_date="2020-01-01")
                fed_funds_raw = qj.fred.get_effective_federal_funds_rate(start_date="2020-01-01")
                macro_prices, volumes = price_panel(["UUP", "GLD", "SPY"], start="2020-01-01", end=END)
                """,
                """
                crypto = pd.concat([
                    series_from_payload("Bitcoin spot (CCXT:BTC/USDT)", btc_raw),
                    series_from_payload("Ethereum spot (CCXT:ETH/USDT)", eth_raw),
                    series_from_payload("Bitcoin spot (CoinGecko)", gecko_raw),
                ], axis=1).dropna(how="all")
                context = pd.concat([
                    series_from_payload("CBOE Volatility Index (VIX)", vix_raw),
                    series_from_payload("10-Year Treasury Yield (FRED:DGS10)", ten_y_raw),
                    series_from_payload("Effective Fed Funds Rate (FRED:FEDFUNDS)", fed_funds_raw),
                ], axis=1).dropna(how="all")
                funding = pd.DataFrame(as_rows(funding_raw))
                open_interest = pd.DataFrame(as_rows(open_interest_raw))
                """,
                """
                if crypto.empty:
                    raise RuntimeError("No crypto spot data returned")
                crypto_ret = crypto.pct_change()
                macro_ret = macro_prices.pct_change()
                joined = crypto_ret.join(macro_ret, how="inner").dropna()
                corr = joined.tail(252).corr().loc[crypto.columns, macro_prices.columns]
                display(pd.Series({"crypto_rows": len(crypto), "funding_rows": len(funding), "open_interest_rows": len(open_interest), "macro_context_rows": len(context)}))
                display(corr)
                crypto.div(crypto.iloc[0]).tail(756).plot(title="Crypto spot feeds normalized")
                plt.ylabel("index")
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "94_news_filings_price_reaction_panel.ipynb",
            "News Filings Price Reaction Panel",
            "Multi-source news and disclosure workflow",
            "Combines financial news, SEC filings, earnings dates, insider transactions and adjusted prices into an issuer event-reaction panel.",
            [
                "qj.tiingo.get_news",
                "qj.finnhub.get_company_news",
                "qj.sec.get_company_filings",
                "qj.sec.get_insider_transactions",
                "qj.fmp.get_earnings_calendar",
                "qj.eod.get_historical_prices",
            ],
            [
                """
                symbol = "AAPL"
                peer_symbols = ["AAPL", "MSFT", "NVDA", "GOOGL"]
                tiingo_news_raw = qj.tiingo.get_news(tickers=symbol, startDate="2024-01-01", endDate=END)
                finnhub_news_raw = qj.finnhub.get_company_news(symbol=symbol, from_date="2024-01-01", to_date=END)
                filings_raw = qj.sec.get_company_filings(symbol=symbol, limit=30)
                insiders_raw = qj.sec.get_insider_transactions(symbol=symbol, limit=100)
                earnings_raw = qj.fmp.get_earnings_calendar(from_date="2024-01-01", to_date=END)
                prices, volumes = price_panel(peer_symbols, start="2023-01-01", end=END)
                """,
                """
                events = []
                for source, payload in {
                    "Tiingo news": tiingo_news_raw,
                    "Finnhub company news": finnhub_news_raw,
                    "SEC filings": filings_raw,
                    "SEC insider transactions": insiders_raw,
                    "FMP earnings calendar": earnings_raw,
                }.items():
                    for row in as_rows(payload):
                        raw_date = row.get("date") or row.get("datetime") or row.get("publishedDate") or row.get("filingDate") or row.get("transactionDate")
                        if isinstance(raw_date, (int, float)):
                            date = pd.to_datetime(raw_date, errors="coerce", unit="s")
                        else:
                            date = pd.to_datetime(raw_date, errors="coerce")
                        if pd.notna(date):
                            events.append({"source": source, "event_date": date.normalize(), "raw": row})
                events = pd.DataFrame(events)
                if events.empty:
                    raise RuntimeError("No issuer event rows returned")
                """,
                """
                px = prices[symbol].dropna()
                reaction_rows = []
                for row in events.itertuples():
                    pos = px.index.searchsorted(row.event_date)
                    if pos > 1 and pos + 5 < len(px):
                        reaction_rows.append({
                            "source": row.source,
                            "event_date": row.event_date,
                            "reaction_1d": px.iloc[pos + 1] / px.iloc[pos] - 1,
                            "reaction_5d": px.iloc[pos + 5] / px.iloc[pos] - 1,
                        })
                reactions = pd.DataFrame(reaction_rows)
                source_summary = reactions.groupby("source")[["reaction_1d", "reaction_5d"]].median().sort_values("reaction_5d")
                display(pd.Series({"event_rows": len(events), "reaction_rows": len(reactions), "peer_price_columns": prices.shape[1]}))
                display(source_summary)
                source_summary.plot(kind="barh", title="Median AAPL price reaction by public event source")
                plt.xlabel("return")
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "95_spread_liquidity_monitoring_packet.ipynb",
            "Spread and Liquidity Monitoring Packet",
            "Market microstructure workflow",
            "Builds spread, liquidity, slippage and capacity diagnostics from intraday prices, adjusted OHLCV, volume, short interest and volatility context.",
            [
                "qj.eod.get_intraday_prices",
                "qj.eod.get_historical_prices",
                "qj.fmp.get_live_prices",
                "qj.finra.get_short_interest",
                "qj.cboe.get_vix_data",
            ],
            [
                """
                symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
                intraday_raw = {symbol: qj.eod.get_intraday_prices(symbol=symbol, interval="5m", from_date=START, to_date=END) for symbol in symbols[:3]}
                live_raw = qj.fmp.get_live_prices(symbol=",".join(symbols))
                short_raw = {symbol: qj.finra.get_short_interest(symbol=symbol) for symbol in symbols[:4]}
                vix_raw = qj.cboe.get_vix_data(start_date="2023-01-01", end_date=END)
                prices, volumes = price_panel(symbols, start="2023-01-01", end=END)
                """,
                """
                ret = returns(prices)
                adv = dollar_adv(prices, volumes).iloc[-1]
                realized_vol = ret.tail(63).std() * np.sqrt(252)
                amihud = (ret.abs() / (prices * volumes).replace(0, np.nan)).tail(63).mean()
                short_rows = pd.Series({symbol: len(as_rows(payload)) for symbol, payload in short_raw.items()}, name="short_feed_rows")
                spread_proxy_bps = (realized_vol.rank(pct=True) * 12 + (1 / adv.rank(pct=True)).replace(np.inf, np.nan) * 4).rename("spread_proxy_bps")
                liquidity = pd.DataFrame({
                    "adv_usd": adv,
                    "realized_vol_63d": realized_vol,
                    "amihud_proxy": amihud,
                    "spread_proxy_bps": spread_proxy_bps,
                    "sl_to_adv_ratio": spread_proxy_bps / adv.rank(pct=True),
                    "short_feed_rows": short_rows,
                }).sort_values("sl_to_adv_ratio", ascending=False)
                """,
                """
                order_notional = 25_000_000
                liquidity["days_at_5pct_adv"] = order_notional / (liquidity["adv_usd"] * 0.05)
                liquidity["slippage_cost_bps"] = liquidity["spread_proxy_bps"] * np.sqrt(np.maximum(liquidity["days_at_5pct_adv"], 0.1))
                display(pd.Series({"live_price_rows": len(as_rows(live_raw)), "vix_rows": len(as_rows(vix_raw)), "intraday_feeds": len(intraday_raw)}))
                display(liquidity)
                liquidity[["spread_proxy_bps", "sl_to_adv_ratio", "days_at_5pct_adv", "slippage_cost_bps"]].plot(kind="bar", subplots=True, layout=(2, 2), figsize=(14, 7), title="Spread and liquidity diagnostics")
                plt.tight_layout()
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "96_earnings_revisions_price_momentum.ipynb",
            "Earnings Revisions and Price Momentum",
            "Forward estimate workflow",
            "Ranks forward EPS and revenue revisions, breadth, dispersion, acceleration and price momentum across a peer set.",
            [
                "qj.fmp.get_analyst_estimates",
                "qj.fmp.get_earnings_surprises",
                "qj.fmp.get_price_target_summary",
                "qj.fmp.get_financial_ratios_ttm",
                "qj.eod.get_historical_prices",
            ],
            [
                """
                symbols = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "AMD"]
                estimates_raw = {symbol: qj.fmp.get_analyst_estimates(symbol=symbol, period="annual", limit=12) for symbol in symbols}
                surprises_raw = {symbol: qj.fmp.get_earnings_surprises(symbol=symbol) for symbol in symbols}
                targets_raw = {symbol: qj.fmp.get_price_target_summary(symbol=symbol) for symbol in symbols}
                ratios_raw = {symbol: qj.fmp.get_financial_ratios_ttm(symbol=symbol) for symbol in symbols}
                prices, volumes = price_panel(symbols, start="2023-01-01", end=END)
                """,
                """
                rows = []
                for symbol in symbols:
                    estimates = pd.DataFrame(as_rows(estimates_raw[symbol]))
                    numeric = estimates.select_dtypes(include="number")
                    eps_path = numeric.mean(axis=1).head(3).to_numpy() if not numeric.empty else np.array([])
                    if len(eps_path) < 3:
                        eps_path = np.array([np.nan, np.nan, np.nan])
                    surprise_rows = pd.DataFrame(as_rows(surprises_raw[symbol]))
                    target_rows = pd.DataFrame(as_rows(targets_raw[symbol]))
                    ratio_row = pd.DataFrame(as_rows(ratios_raw[symbol])).head(1)
                    rows.append({
                        "symbol": symbol,
                        "cfy_estimate": eps_path[0],
                        "nfy_estimate": eps_path[1],
                        "fy2_estimate": eps_path[2],
                        "revision_breadth": numeric.diff().gt(0).mean().mean() if not numeric.empty else np.nan,
                        "revision_dispersion": numeric.std(axis=1).mean() / numeric.mean(axis=1).abs().mean() if not numeric.empty else np.nan,
                        "surprise_rows": len(surprise_rows),
                        "price_target_rows": len(target_rows),
                        "pe_ttm": pd.to_numeric(ratio_row.get("peRatioTTM", pd.Series([np.nan])).iloc[0], errors="coerce") if not ratio_row.empty else np.nan,
                    })
                revisions = pd.DataFrame(rows).set_index("symbol")
                """,
                """
                momentum_12_1 = prices.pct_change(252).iloc[-1] - prices.pct_change(21).iloc[-1]
                momentum_3m = prices.pct_change(63).iloc[-1]
                revisions["eps_acceleration"] = revisions["fy2_estimate"] - revisions["cfy_estimate"]
                revisions["price_momentum_12_1"] = momentum_12_1.reindex(revisions.index)
                revisions["price_momentum_3m"] = momentum_3m.reindex(revisions.index)
                revisions["composite_revision_momentum"] = (
                    revisions["revision_breadth"].rank(pct=True)
                    - revisions["revision_dispersion"].rank(pct=True)
                    + revisions["eps_acceleration"].rank(pct=True)
                    + revisions["price_momentum_12_1"].rank(pct=True)
                )
                display(revisions.sort_values("composite_revision_momentum", ascending=False))
                revisions[["revision_breadth", "revision_dispersion", "eps_acceleration", "price_momentum_12_1"]].plot(kind="bar", subplots=True, layout=(2, 2), figsize=(14, 7), title="Revisions and momentum inputs")
                plt.tight_layout()
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "97_roic_wacc_cash_conversion_quality.ipynb",
            "ROIC WACC and Cash Conversion Quality",
            "Fundamental quality workflow",
            "Uses statements, ratios, key metrics, market rates and profile data to compute ROIC-WACC spread, reinvestment and cash conversion quality.",
            [
                "qj.fmp.get_income_statement",
                "qj.fmp.get_balance_sheet_statement",
                "qj.fmp.get_cash_flow_statement",
                "qj.fmp.get_key_metrics_ttm",
                "qj.fmp.get_financial_ratios_ttm",
                "qj.fred.get_treasury_10y",
                "qj.eod.get_historical_prices",
            ],
            [
                """
                symbols = ["MSFT", "AAPL", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "COST"]
                income_raw = {symbol: qj.fmp.get_income_statement(symbol=symbol, period="annual", limit=5) for symbol in symbols}
                balance_raw = {symbol: qj.fmp.get_balance_sheet_statement(symbol=symbol, period="annual", limit=5) for symbol in symbols}
                cash_flow_raw = {symbol: qj.fmp.get_cash_flow_statement(symbol=symbol, period="annual", limit=5) for symbol in symbols}
                key_metrics_raw = {symbol: qj.fmp.get_key_metrics_ttm(symbol=symbol) for symbol in symbols}
                ratios_raw = {symbol: qj.fmp.get_financial_ratios_ttm(symbol=symbol) for symbol in symbols}
                ten_y_raw = qj.fred.get_treasury_10y(start_date="2020-01-01")
                prices, volumes = price_panel(symbols, start="2021-01-01", end=END)
                """,
                """
                def first_row(payload: Any) -> dict[str, Any]:
                    rows = as_rows(payload)
                    return rows[0] if rows and isinstance(rows[0], dict) else {}

                quality_rows = []
                for symbol in symbols:
                    income = first_row(income_raw[symbol])
                    balance = first_row(balance_raw[symbol])
                    cash_flow = first_row(cash_flow_raw[symbol])
                    metrics = first_row(key_metrics_raw[symbol])
                    ratios = first_row(ratios_raw[symbol])
                    nopat = pd.to_numeric(income.get("operatingIncome") or income.get("ebit"), errors="coerce") * 0.79
                    invested_capital = pd.to_numeric(balance.get("totalDebt"), errors="coerce") + pd.to_numeric(balance.get("totalStockholdersEquity"), errors="coerce") - pd.to_numeric(balance.get("cashAndCashEquivalents"), errors="coerce")
                    roic = nopat / invested_capital if invested_capital else np.nan
                    fcf = pd.to_numeric(cash_flow.get("freeCashFlow"), errors="coerce")
                    net_income = pd.to_numeric(income.get("netIncome"), errors="coerce")
                    quality_rows.append({
                        "symbol": symbol,
                        "roic": roic,
                        "wacc_proxy": pd.to_numeric(metrics.get("weightedAverageCostOfCapital"), errors="coerce") if metrics else np.nan,
                        "fcf_conversion": fcf / net_income if net_income else np.nan,
                        "gross_margin_ttm": pd.to_numeric(ratios.get("grossProfitMarginTTM"), errors="coerce"),
                        "debt_to_equity_ttm": pd.to_numeric(ratios.get("debtEquityRatioTTM"), errors="coerce"),
                    })
                quality = pd.DataFrame(quality_rows).set_index("symbol")
                """,
                """
                if quality["wacc_proxy"].isna().all():
                    rate_rows = pd.DataFrame(as_rows(ten_y_raw))
                    risk_free = pd.to_numeric(rate_rows.select_dtypes(include="number").stack(), errors="coerce").dropna().tail(1).mean() / 100
                    quality["wacc_proxy"] = risk_free + 0.055
                quality["roic_wacc_spread"] = quality["roic"] - quality["wacc_proxy"]
                quality["quality_score"] = quality["roic_wacc_spread"].rank(pct=True) + quality["fcf_conversion"].rank(pct=True) + quality["gross_margin_ttm"].rank(pct=True)
                display(quality.sort_values("quality_score", ascending=False))
                quality[["roic", "wacc_proxy", "roic_wacc_spread", "fcf_conversion"]].plot(kind="bar", subplots=True, layout=(2, 2), figsize=(14, 7), title="ROIC, WACC and cash conversion")
                plt.tight_layout()
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "98_oil_curve_regime_instrument_selection.ipynb",
            "Oil Curve Regime and Instrument Selection",
            "Commodity curve workflow",
            "Combines WTI reference data, futures pricing, energy ETFs, COT positioning and petroleum data to classify oil curve regimes and instrument fit.",
            [
                "qj.eod.get_futures_contracts",
                "qj.eod.get_futures_pricing",
                "qj.eia.get_petroleum_prices",
                "qj.cftc.get_cot_summary",
                "qj.eod.get_historical_prices",
            ],
            [
                """
                futures_contracts = qj.eod.get_futures_contracts(exchange="COMM")
                cl_front = qj.eod.get_futures_pricing(symbol="CL1", start_date="2022-01-01", end_date=END)
                cl_second = qj.eod.get_futures_pricing(symbol="CL2", start_date="2022-01-01", end_date=END)
                petroleum_raw = qj.eia.get_petroleum_prices()
                cot_raw = qj.cftc.get_cot_summary(symbol="CL")
                prices, volumes = price_panel(["USO", "XLE", "XOP", "SPY"], start="2022-01-01", end=END)
                """,
                """
                def close_series(name: str, payload: Any) -> pd.Series:
                    frame = pd.DataFrame(as_rows(payload))
                    if frame.empty:
                        return pd.Series(dtype=float, name=name)
                    date_col = next((col for col in frame.columns if "date" in str(col).lower()), frame.columns[0])
                    value_col = "close" if "close" in frame.columns else frame.select_dtypes(include="number").columns[-1]
                    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce")
                    frame[name] = pd.to_numeric(frame[value_col], errors="coerce")
                    return frame.dropna(subset=["date", name]).set_index("date")[name].sort_index()

                curve = pd.concat([close_series("CL1 front month", cl_front), close_series("CL2 second month", cl_second)], axis=1).dropna(how="all")
                """,
                """
                if curve.empty:
                    raise RuntimeError("No oil futures curve data returned")
                curve["front_second_spread"] = curve["CL1 front month"] - curve["CL2 second month"]
                curve["roll_yield_proxy"] = curve["front_second_spread"] / curve["CL1 front month"]
                regime = np.where(curve["front_second_spread"] > 0, "backwardation", "contango")
                instrument = pd.DataFrame({
                    "return_63d": prices.pct_change(63).iloc[-1],
                    "volatility_63d": returns(prices).tail(63).std() * np.sqrt(252),
                    "oil_beta_proxy": returns(prices).tail(252).corrwith(curve["CL1 front month"].pct_change().reindex(prices.index)),
                })
                display(pd.Series({"contracts_rows": len(as_rows(futures_contracts)), "petroleum_rows": len(as_rows(petroleum_raw)), "cot_rows": len(as_rows(cot_raw)), "latest_regime": regime[-1]}))
                display(instrument.sort_values("oil_beta_proxy", ascending=False))
                curve[["CL1 front month", "CL2 second month", "front_second_spread"]].tail(504).plot(title="WTI curve and front-second spread")
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "99_vixy_term_structure_macro_regime.ipynb",
            "VIXY Term Structure and Macro Regime",
            "Volatility ETP workflow",
            "Combines VIX, VVIX, SKEW, VIX futures term structure, VIXY price history and macro rates to quantify carry and regime state.",
            [
                "qj.cboe.get_vix_data",
                "qj.cboe.get_vvix_data",
                "qj.cboe.get_skew_index_data",
                "qj.cboe.get_vix_term_structure",
                "qj.eod.get_historical_prices",
                "qj.fred.get_treasury_10y",
            ],
            [
                """
                vix_raw = qj.cboe.get_vix_data(start_date="2020-01-01", end_date=END)
                vvix_raw = qj.cboe.get_vvix_data(start_date="2020-01-01", end_date=END)
                skew_raw = qj.cboe.get_skew_index_data(start_date="2020-01-01", end_date=END)
                term_raw = qj.cboe.get_vix_term_structure()
                ten_y_raw = qj.fred.get_treasury_10y(start_date="2020-01-01")
                prices, volumes = price_panel(["VIXY", "SPY", "TLT"], start="2020-01-01", end=END)
                """,
                """
                def series_from_rows(name: str, payload: Any) -> pd.Series:
                    frame = pd.DataFrame(as_rows(payload))
                    if frame.empty:
                        return pd.Series(dtype=float, name=name)
                    date_col = next((col for col in frame.columns if "date" in str(col).lower()), frame.columns[0])
                    value_col = "close" if "close" in frame.columns else frame.select_dtypes(include="number").columns[-1]
                    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce")
                    frame[name] = pd.to_numeric(frame[value_col], errors="coerce")
                    return frame.dropna(subset=["date", name]).set_index("date")[name].sort_index()

                vol = pd.concat([series_from_rows("VIX", vix_raw), series_from_rows("VVIX", vvix_raw), series_from_rows("SKEW", skew_raw)], axis=1).dropna(how="all")
                term = pd.DataFrame(as_rows(term_raw))
                """,
                """
                ret = returns(prices)
                carry_proxy = prices["VIXY"].pct_change(21) - prices["SPY"].pct_change(21)
                regime = pd.DataFrame({
                    "vixy_return_21d": prices["VIXY"].pct_change(21),
                    "spy_return_21d": prices["SPY"].pct_change(21),
                    "carry_proxy": carry_proxy,
                    "vix_level": vol["VIX"].reindex(prices.index).ffill() if "VIX" in vol else np.nan,
                    "vixy_volatility_63d": ret["VIXY"].rolling(63).std() * np.sqrt(252),
                }).dropna(how="all")
                display(pd.Series({"vix_rows": len(as_rows(vix_raw)), "vvix_rows": len(as_rows(vvix_raw)), "skew_rows": len(as_rows(skew_raw)), "term_rows": len(term), "rate_rows": len(as_rows(ten_y_raw))}))
                display(regime.tail())
                regime[["vixy_return_21d", "carry_proxy", "vixy_volatility_63d"]].tail(504).plot(title="VIXY carry and volatility regime")
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "100_credit_hy_treasury_spread_regime.ipynb",
            "Credit High Yield Treasury Spread Regime",
            "Credit and macro workflow",
            "Combines high-yield spreads, Treasury rates, credit ETFs, equity prices and VIX into spread, duration and risk-regime diagnostics.",
            [
                "qj.fred.get_fred_data_series_by_id",
                "qj.fred.get_treasury_2y",
                "qj.fred.get_treasury_10y",
                "qj.cboe.get_vix_data",
                "qj.eod.get_historical_prices",
            ],
            [
                """
                macro_raw = {
                    "High Yield OAS (FRED:BAMLH0A0HYM2)": qj.fred.get_fred_data_series_by_id(series_id="BAMLH0A0HYM2", start="2020-01-01"),
                    "Investment Grade OAS (FRED:BAMLC0A0CM)": qj.fred.get_fred_data_series_by_id(series_id="BAMLC0A0CM", start="2020-01-01"),
                    "2-Year Treasury Yield (FRED:DGS2)": qj.fred.get_treasury_2y(start_date="2020-01-01"),
                    "10-Year Treasury Yield (FRED:DGS10)": qj.fred.get_treasury_10y(start_date="2020-01-01"),
                }
                vix_raw = qj.cboe.get_vix_data(start_date="2020-01-01", end_date=END)
                prices, volumes = price_panel(["HYG", "LQD", "TLT", "IEF", "SPY"], start="2020-01-01", end=END)
                """,
                """
                def series_from_payload(name: str, payload: Any) -> pd.Series:
                    frame = pd.DataFrame(as_rows(payload))
                    if frame.empty:
                        return pd.Series(dtype=float, name=name)
                    date_col = next((col for col in frame.columns if "date" in str(col).lower()), frame.columns[0])
                    value_col = "value" if "value" in frame.columns else frame.select_dtypes(include="number").columns[-1]
                    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce")
                    frame[name] = pd.to_numeric(frame[value_col], errors="coerce")
                    return frame.dropna(subset=["date", name]).set_index("date")[name].sort_index()

                macro = pd.concat([series_from_payload(name, payload) for name, payload in macro_raw.items()], axis=1).dropna(how="all")
                vix = series_from_payload("CBOE Volatility Index (VIX)", vix_raw)
                """,
                """
                if macro.empty:
                    raise RuntimeError("No credit spread macro data returned")
                ret = returns(prices)
                hy_col = "High Yield OAS (FRED:BAMLH0A0HYM2)"
                ig_col = "Investment Grade OAS (FRED:BAMLC0A0CM)"
                spread = macro[[hy_col, ig_col]].dropna(how="all")
                spread["HY_to_IG_spread_ratio"] = spread[hy_col] / spread[ig_col]
                credit_panel = pd.DataFrame({
                    "hyg_return_63d": prices["HYG"].pct_change(63),
                    "hyg_lqd_spread_trade": prices["HYG"].pct_change(63) - prices["LQD"].pct_change(63),
                    "duration_trade_tlt_ief": prices["TLT"].pct_change(63) - prices["IEF"].pct_change(63),
                    "vix": vix.reindex(prices.index).ffill(),
                }).dropna(how="all")
                display(spread.tail())
                display((ret.tail(252).corr()).loc[["HYG", "LQD", "TLT", "SPY"], ["HYG", "LQD", "TLT", "SPY"]])
                spread[[hy_col, ig_col, "HY_to_IG_spread_ratio"]].tail(504).plot(title="Credit spread and HY/IG ratio")
                plt.show()
                """,
            ],
        ),
        NotebookSpec(
            "101_consumer_dispersion_not_the_trade.ipynb",
            "Consumer Dispersion Not The Trade",
            "Consumer macro and equity workflow",
            "Separates consumer exposure into discretionary, staples, retail, travel and balance-sheet-quality buckets using macro data, sector ETFs, fundamentals and prices.",
            [
                "qj.fred.get_fred_data_series_by_id",
                "qj.fred.get_cpi",
                "qj.fred.get_effective_federal_funds_rate",
                "qj.fmp.get_financial_ratios_ttm",
                "qj.fmp.get_company_profile",
                "qj.eod.get_historical_prices",
            ],
            [
                """
                symbols = ["XLY", "XLP", "WMT", "COST", "HD", "LOW", "MCD", "SBUX", "TSLA", "AMZN", "NKE"]
                macro_raw = {
                    "Retail Sales (FRED:RSAFS)": qj.fred.get_fred_data_series_by_id(series_id="RSAFS", start="2018-01-01"),
                    "Consumer Sentiment (FRED:UMCSENT)": qj.fred.get_fred_data_series_by_id(series_id="UMCSENT", start="2018-01-01"),
                    "Consumer Credit (FRED:TOTALSL)": qj.fred.get_fred_data_series_by_id(series_id="TOTALSL", start="2018-01-01"),
                    "CPI Urban Consumers (FRED:CPIAUCSL)": qj.fred.get_cpi(start_date="2018-01-01"),
                    "Effective Fed Funds Rate (FRED:FEDFUNDS)": qj.fred.get_effective_federal_funds_rate(start_date="2018-01-01"),
                }
                ratios_raw = {symbol: qj.fmp.get_financial_ratios_ttm(symbol=symbol) for symbol in symbols[2:]}
                profiles_raw = {symbol: qj.fmp.get_company_profile(symbol=symbol) for symbol in symbols[2:]}
                prices, volumes = price_panel(symbols, start="2018-01-01", end=END)
                """,
                """
                def first_row(payload: Any) -> dict[str, Any]:
                    rows = as_rows(payload)
                    return rows[0] if rows and isinstance(rows[0], dict) else {}

                ret = returns(prices)
                rows = []
                for symbol in symbols[2:]:
                    ratios = first_row(ratios_raw[symbol])
                    profile = first_row(profiles_raw[symbol])
                    rows.append({
                        "symbol": symbol,
                        "industry": profile.get("industry") or profile.get("sector"),
                        "return_126d": prices[symbol].pct_change(126).iloc[-1],
                        "volatility_63d": ret[symbol].tail(63).std() * np.sqrt(252),
                        "gross_margin_ttm": pd.to_numeric(ratios.get("grossProfitMarginTTM"), errors="coerce"),
                        "current_ratio_ttm": pd.to_numeric(ratios.get("currentRatioTTM"), errors="coerce"),
                        "debt_to_equity_ttm": pd.to_numeric(ratios.get("debtEquityRatioTTM"), errors="coerce"),
                    })
                consumer = pd.DataFrame(rows).set_index("symbol")
                """,
                """
                consumer["resilience_score"] = (
                    consumer["gross_margin_ttm"].rank(pct=True)
                    + consumer["current_ratio_ttm"].rank(pct=True)
                    - consumer["debt_to_equity_ttm"].rank(pct=True)
                    + consumer["return_126d"].rank(pct=True)
                    - consumer["volatility_63d"].rank(pct=True)
                )
                etf_spread = prices["XLP"].pct_change(126) - prices["XLY"].pct_change(126)
                macro_rows = pd.Series({name: len(as_rows(payload)) for name, payload in macro_raw.items()})
                display(macro_rows.rename("macro_rows"))
                display(consumer.sort_values("resilience_score", ascending=False))
                consumer[["return_126d", "gross_margin_ttm", "current_ratio_ttm", "debt_to_equity_ttm", "resilience_score"]].plot(kind="bar", subplots=True, layout=(2, 3), figsize=(15, 8), title="Consumer dispersion inputs")
                plt.tight_layout()
                plt.show()
                """,
            ],
        ),
    ]
