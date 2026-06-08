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
    ]
