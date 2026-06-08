"""Build the flat `_candidates/` example notebook catalog.

The example catalog is intentionally flat: every workflow notebook lives
directly under `_candidates/<number>_<slug>.ipynb`. Existing core notebooks are
copied in, and institutional workflows are generated as
self-contained source notebooks with real QuantJourney SDK calls plus local
pandas / numpy analytics.
"""

from __future__ import annotations

import ast
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "_candidates"


EXISTING_CORE = [
    "01_authentication_methods.ipynb",
    "02_market_data_basics.ipynb",
    "03_economic_data_macro.ipynb",
    "04_fundamental_analysis.ipynb",
    "05_technical_analysis.ipynb",
    "06_portfolio_analysis.ipynb",
    "07_crypto_ccxt.ipynb",
    "08_cboe_vix.ipynb",
    "09_multpl_valuation.ipynb",
    "10_cftc_cot.ipynb",
    "11_sec_filings.ipynb",
    "12_finra_short_interest.ipynb",
    "13_openfigi.ipynb",
]


EXISTING_BUY_SIDE: list[str] = []


PRESERVED_SOURCE_CANDIDATES = [
    (
        "70_macro_regime_cot_cross_asset.ipynb",
        "Multi-source macro / positioning / allocation",
        "Fuses macro indicators, CFTC positioning, cross-asset pricing and volatility context into a tactical allocation view.",
    ),
    (
        "71_congress_13f_smart_money_mosaic.ipynb",
        "Multi-source regulatory / alt-data / flow analysis",
        "Combines congressional trades, institutional holdings and price reaction into a smart-money overlay signal.",
    ),
    (
        "72_crowding_liquidity_capacity_stress.ipynb",
        "Multi-source crowding / liquidity / risk",
        "Combines ownership, ADV, short-interest and stress context into a crowding and capacity screen.",
    ),
    (
        "73_conditioned_earnings_pead_multi_source.ipynb",
        "Multi-source event study / alpha research",
        "Conditions post-earnings drift analysis on earnings surprises, liquidity, short-interest and macro context.",
    ),
    (
        "74_factor_macro_risk_shock_transmission.ipynb",
        "Multi-source risk model / stress testing",
        "Maps macro shocks through factor exposures and volatility context into stress P&L diagnostics.",
    ),
    (
        "75_public_signals_book_intelligence.ipynb",
        "Multi-source daily intelligence / decision support",
        "Overlays public signals across earnings, congress, 13F, macro, COT and volatility for a book-level watchlist.",
    ),
]
PRESERVED_SOURCE_CANDIDATE_NAMES = {name for name, _, _ in PRESERVED_SOURCE_CANDIDATES}


IMPORTS_AND_PLOT_STYLE = r'''
import os
import math
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quantjourney.sdk import QuantJourneyAPI

plt.style.use("default")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
'''


API_CLIENT_SETUP = r'''
qj = QuantJourneyAPI(api_key=os.environ["QJ_API_KEY"])

START = os.getenv("QJ_EXAMPLE_START", "2020-01-01")
END = os.getenv("QJ_EXAMPLE_END") or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
'''


RESPONSE_HELPERS = r'''
def unwrap(payload: Any) -> Any:
    """Return the useful data value from common QuantJourney response shapes."""
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if isinstance(payload, dict) and "value" in payload:
        return payload["value"]
    return payload


def as_rows(payload: Any) -> list[dict[str, Any]]:
    value = unwrap(payload)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("rows", "data", "items", "prices", "results"):
            if isinstance(value.get(key), list):
                return value[key]
        return [value]
    return []
'''


MARKET_DATA_HELPERS = r'''
def price_frame(symbol: str, start: str = START, end: str = END) -> pd.DataFrame:
    payload = qj.eod.get_historical_prices(symbol=symbol, start_date=start, end_date=end)
    rows = as_rows(payload)
    if not rows and isinstance(unwrap(payload), dict):
        value = unwrap(payload)
        rows = value.get(symbol) or value.get(symbol.upper()) or []
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No price data returned for {symbol}")
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "adjusted_close" in df and df["adjusted_close"].notna().any():
        df["price"] = df["adjusted_close"].fillna(df["close"])
    else:
        df["price"] = df["close"]
    if "volume" not in df:
        df["volume"] = np.nan
    return df.dropna(subset=["price"]).sort_values("date").set_index("date")


def price_panel(symbols: list[str], start: str = START, end: str = END) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = {}
    volumes = {}
    for symbol in symbols:
        df = price_frame(symbol, start=start, end=end)
        prices[symbol] = df["price"]
        volumes[symbol] = df["volume"]
    return pd.DataFrame(prices).dropna(how="all"), pd.DataFrame(volumes).reindex(pd.DataFrame(prices).index)


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")


def dollar_adv(prices: pd.DataFrame, volumes: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    return (prices * volumes).rolling(window).mean()
'''


PORTFOLIO_HELPERS = r'''
def max_drawdown(nav: pd.Series) -> float:
    drawdown = nav / nav.cummax() - 1
    return float(drawdown.min())


def performance_stats(ret: pd.Series) -> pd.Series:
    ret = ret.dropna()
    nav = (1 + ret).cumprod()
    ann_ret = nav.iloc[-1] ** (252 / len(ret)) - 1 if len(ret) and nav.iloc[-1] > 0 else np.nan
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol and np.isfinite(ann_vol) else np.nan
    return pd.Series({
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(nav) if len(nav) else np.nan,
        "total_return": nav.iloc[-1] - 1 if len(nav) else np.nan,
    })


def inverse_vol_weights(ret: pd.DataFrame) -> pd.Series:
    vol = ret.std().replace(0, np.nan)
    inv = 1 / vol
    return (inv / inv.sum()).fillna(0)


def min_variance_weights(ret: pd.DataFrame, ridge: float = 1e-4) -> pd.Series:
    cov = ret.cov().fillna(0).to_numpy() * 252
    cov = cov + np.eye(cov.shape[0]) * ridge
    inv = np.linalg.pinv(cov)
    raw = inv @ np.ones(cov.shape[0])
    raw = np.maximum(raw, 0)
    if raw.sum() == 0:
        raw = np.ones(cov.shape[0])
    return pd.Series(raw / raw.sum(), index=ret.columns)


def portfolio_returns(ret: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned = ret[weights.index].dropna()
    return aligned @ weights.reindex(aligned.columns).fillna(0)


def risk_contribution(ret: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned = ret[weights.index].dropna()
    cov = aligned.cov() * 252
    w = weights.reindex(cov.columns).fillna(0).to_numpy()
    port_var = float(w @ cov.to_numpy() @ w)
    if port_var <= 0:
        return pd.Series(0.0, index=cov.columns)
    contrib = w * (cov.to_numpy() @ w) / port_var
    return pd.Series(contrib, index=cov.columns)
'''


FACTOR_HELPERS = r'''
def rolling_betas(y: pd.Series, x: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    data = pd.concat([y.rename("asset"), x], axis=1).dropna()
    rows = []
    for i in range(window, len(data)):
        chunk = data.iloc[i - window:i]
        yy = chunk["asset"].to_numpy()
        xx = np.column_stack([np.ones(len(chunk)), chunk[x.columns].to_numpy()])
        beta = np.linalg.lstsq(xx, yy, rcond=None)[0][1:]
        rows.append(dict(date=data.index[i], **{col: beta[j] for j, col in enumerate(x.columns)}))
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=x.columns)


def zscore(s: pd.Series, window: int = 252) -> pd.Series:
    return (s - s.rolling(window).mean()) / s.rolling(window).std()
'''


PLOT_HELPERS = r'''
def plot_nav(ret_map: dict[str, pd.Series], title: str) -> None:
    fig, ax = plt.subplots()
    for label, ret in ret_map.items():
        nav = (1 + ret.dropna()).cumprod()
        ax.plot(nav.index, nav, label=label)
    ax.set_title(title)
    ax.legend()
    plt.show()
'''


@dataclass(frozen=True)
class NotebookSpec:
    filename: str
    title: str
    category: str
    summary: str
    data_calls: list[str]
    cells: list[str]
    mirror_to_advanced: bool = False


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(keepends=True)}


EMOJI_REPLACEMENTS = {
    "\u2713": "OK",
    "\u2705": "OK",
    "\u26a0\ufe0f": "Warning",
    "\u26a0": "Warning",
    "\U0001f534": "",
    "\U0001f7e2": "",
    "\U0001f7e1": "",
    "\U0001f4ca": "",
    "\U0001f4c8": "",
    "\U0001f4c9": "",
    "\U0001f4a1": "",
    "\U0001f680": "",
    "\U0001f525": "",
    "\u2b50": "",
}


def strip_icons(source: str) -> str:
    for old, new in EMOJI_REPLACEMENTS.items():
        source = source.replace(old, new)
    return source


def normalize_core_setup_cells(nb: dict, path: Path) -> None:
    if not path.name[:2].isdigit() or not (2 <= int(path.name[:2]) <= 10):
        return
    if len(nb.get("cells", [])) < 2 or nb["cells"][1].get("cell_type") != "code":
        return
    source = "".join(nb["cells"][1].get("source", []))
    if "QuantJourneyAPI" not in source or "sys.path.insert" not in source:
        return

    import_lines: list[str] = []
    seen: set[str] = set()
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("import sys") or stripped.startswith("sys.path.insert"):
            continue
        if stripped.startswith("from quantjourney.sdk import QuantJourneyAPI"):
            continue
        if stripped.startswith("API_KEY") or stripped.startswith("qj =") or stripped.startswith("print("):
            continue
        if stripped not in seen:
            import_lines.append(stripped)
            seen.add(stripped)

    client = """
    from quantjourney.sdk import QuantJourneyAPI

    qj = QuantJourneyAPI(api_key=os.environ["QJ_API_KEY"])

    print("Connected to QuantJourney API")
    """
    replacement = [
        md("## Imports and Plot Style"),
        code("\n".join(import_lines)),
        md("## QuantJourney Client"),
        code(client),
    ]
    nb["cells"] = [nb["cells"][0], *replacement, *nb["cells"][2:]]


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Call):
        return _root_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    return None


def _contains_qj_call(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _root_name(child.func) == "qj":
            return True
    return False


class StrictExampleTransformer(ast.NodeTransformer):
    """Normalize generated examples to strict API calls with no helper fallback."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST | None:
        if node.name == "safe_call":
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "safe_call" and len(node.args) >= 2:
            target = node.args[1]
            args = list(node.args[2:])
            keywords = list(node.keywords)
            if isinstance(target, ast.Lambda):
                return ast.copy_location(ast.Call(func=target, args=[], keywords=[]), node)
            return ast.copy_location(ast.Call(func=target, args=args, keywords=keywords), node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_env"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "QuantJourneyAPI"
        ):
            api_key = ast.Subscript(
                value=ast.Attribute(value=ast.Name(id="os", ctx=ast.Load()), attr="environ", ctx=ast.Load()),
                slice=ast.Constant(value="QJ_API_KEY"),
                ctx=ast.Load(),
            )
            return ast.copy_location(
                ast.Call(func=ast.Name(id="QuantJourneyAPI", ctx=ast.Load()), args=[], keywords=[ast.keyword(arg="api_key", value=api_key)]),
                node,
            )
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "environ"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "os"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "QJ_API_KEY"
        ):
            return ast.copy_location(
                ast.Subscript(
                    value=ast.Attribute(value=ast.Name(id="os", ctx=ast.Load()), attr="environ", ctx=ast.Load()),
                    slice=ast.Constant(value="QJ_API_KEY"),
                    ctx=ast.Load(),
                ),
                node,
            )
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Or) and any(_contains_qj_call(value) for value in node.values):
            return ast.copy_location(node.values[0], node)
        return node


def normalize_code_source(source: str) -> str:
    tree = ast.parse(source)
    tree = StrictExampleTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def code(text: str) -> dict:
    source = strip_icons(dedent(text).strip()) + "\n"
    source = normalize_code_source(source)
    ast.parse(source)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def sentence_case(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    return stripped[0].lower() + stripped[1:]


def example_intro(title: str, summary: str) -> str:
    return f"""
    # QuantJourney SDK - {title}

    This notebook demonstrates a QuantJourney SDK workflow that {sentence_case(summary).rstrip(".")}.

    It covers:

    - Direct QuantJourney SDK calls for the required market, macro, regulatory or portfolio data
    - Transparent pandas/numpy calculations so research assumptions stay visible
    - Chart-ready output that can be reused in notebooks, reports or API documentation

    ## Prerequisites

    Make sure you have:

    - Access to QuantJourney API (https://api.quantjourney.cloud)
    - `QJ_API_KEY` configured in your environment
    - Tenant access to the connectors used by this example
    """


def uses_any(source: str, names: Iterable[str]) -> bool:
    return any(f"{name}(" in source for name in names)


def setup_cells_for(spec: NotebookSpec) -> list[dict]:
    source = "\n".join(spec.cells)
    cells = [
        md("## Imports and Plot Style"),
        code(IMPORTS_AND_PLOT_STYLE),
        md("## QuantJourney Client"),
        code(API_CLIENT_SETUP),
    ]

    needs_response = uses_any(source, ["unwrap", "as_rows", "price_frame", "price_panel"])
    needs_market = uses_any(source, ["price_frame", "price_panel", "returns", "dollar_adv"])
    needs_portfolio = uses_any(
        source,
        [
            "max_drawdown",
            "performance_stats",
            "inverse_vol_weights",
            "min_variance_weights",
            "portfolio_returns",
            "risk_contribution",
        ],
    )
    needs_factor = uses_any(source, ["rolling_betas", "zscore"])
    needs_plot = uses_any(source, ["plot_nav"])

    if needs_response or needs_market:
        cells.extend([md("## Response Helpers"), code(RESPONSE_HELPERS)])
    if needs_market:
        cells.extend([md("## Market Data Helpers"), code(MARKET_DATA_HELPERS)])
    if needs_portfolio:
        cells.extend([md("## Portfolio and Risk Helpers"), code(PORTFOLIO_HELPERS)])
    if needs_factor:
        cells.extend([md("## Factor and Regime Helpers"), code(FACTOR_HELPERS)])
    if needs_plot:
        cells.extend([md("## Plot Helpers"), code(PLOT_HELPERS)])

    return cells


def notebook(spec: NotebookSpec) -> dict:
    cells = [md(example_intro(spec.title, spec.summary))]
    cells.extend(setup_cells_for(spec))
    cells.extend(code(cell) for cell in spec.cells)
    cells.append(md("""
    ## Notes

    This is an example workflow. In production, tenant scopes, connector allowlists,
    provider metadata, request IDs and audit logs should be retained next to the resulting
    tables or charts.
    """))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


CORE_PRIMITIVE_SPECS = [
    NotebookSpec(
        "14_corporate_actions_pit_adjustments.ipynb",
        "Corporate Actions and Adjustment Semantics",
        "Core data primitive",
        "Checks adjusted price consistency, dividend history and corporate-action evidence around a single equity.",
        [
            "qj.eod.get_historical_prices",
            "qj.fmp.get_dividends_historical",
            "qj.fmp.get_last_dividend",
            "qj.eod.get_shares_stats",
            "qj.fmp.get_company_profile",
        ],
        [
            """
            symbol = "AAPL"
            prices_raw = safe_call("EOD historical adjusted prices", qj.eod.get_historical_prices, symbol=symbol, start_date="2018-01-01", end_date=END)
            dividends_raw = safe_call("FMP historical dividends", qj.fmp.get_dividends_historical, symbol=symbol)
            last_dividend_raw = safe_call("FMP last dividend", qj.fmp.get_last_dividend, symbol=symbol)
            shares_raw = safe_call("EOD shares stats", qj.eod.get_shares_stats, symbol=symbol)
            profile_raw = safe_call("FMP company profile", qj.fmp.get_company_profile, symbol=symbol)
            """,
            """
            prices = pd.DataFrame(as_rows(prices_raw))
            if not prices.empty:
                prices["date"] = pd.to_datetime(prices.get("date"), errors="coerce")
                for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
                    if col in prices:
                        prices[col] = pd.to_numeric(prices[col], errors="coerce")
                prices = prices.dropna(subset=["date"]).sort_values("date").set_index("date")
                if "adjusted_close" in prices and "close" in prices:
                    prices["adjustment_ratio"] = prices["adjusted_close"] / prices["close"]
                    prices["adjustment_gap"] = prices["adjusted_close"] - prices["close"]

            dividends = pd.DataFrame(as_rows(dividends_raw))
            if not dividends.empty:
                date_col = next((col for col in dividends.columns if "date" in str(col).lower()), dividends.columns[0])
                value_col = next((col for col in dividends.columns if "dividend" in str(col).lower() or "adj" in str(col).lower()), None)
                dividends["date"] = pd.to_datetime(dividends[date_col], errors="coerce")
                if value_col:
                    dividends["dividend"] = pd.to_numeric(dividends[value_col], errors="coerce")
                dividends = dividends.dropna(subset=["date"]).sort_values("date")
            """,
            """
            audit = pd.Series({
                "price_rows": len(prices),
                "dividend_rows": len(dividends),
                "last_dividend_available": bool(as_rows(last_dividend_raw)),
                "shares_stats_available": bool(as_rows(shares_raw)),
                "profile_available": bool(as_rows(profile_raw)),
                "has_adjusted_close": "adjusted_close" in prices.columns if not prices.empty else False,
                "adjustment_changes": int(prices["adjustment_ratio"].diff().abs().gt(0.001).sum()) if "adjustment_ratio" in prices else 0,
            })
            display(audit)
            if not prices.empty and {"close", "adjusted_close"}.issubset(prices.columns):
                prices[["close", "adjusted_close"]].dropna().tail(1000).plot(title="Close vs adjusted close")
                plt.ylabel("price")
                plt.show()
            if not dividends.empty and "dividend" in dividends:
                dividends.tail(40).set_index("date")["dividend"].plot(kind="bar", title="Recent dividend events")
                plt.ylabel("cash dividend")
                plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "15_domain_route_discovery_contract.ipynb",
        "Domain Route Discovery and Contract Introspection",
        "Core data primitive",
        "Uses domain discovery, aliases and route descriptions to inspect the governed API contract available to a tenant.",
        [
            "qj.domains.list",
            "qj.domains.tree",
            "qj.domains.aliases",
            "qj.domains.describe",
            "qj.domains.call",
        ],
        [
            """
            domains_list = qj.domains.list()
            domain_tree = qj.domains.tree(scope="effective", include_aliases=True)
            aliases = qj.domains.aliases()
            route_names = [
                "equity.pricing.get_historical_prices",
                "equity.fundamentals.get_financial_ratios_ttm",
                "macro.economic.get_treasury_rates",
                "derivatives.vol.get_vix_data",
                "reference.identifiers.get_figi_data",
            ]
            descriptions = {route: qj.domains.describe(route=route) for route in route_names}
            """,
            """
            rows = []
            for route, payload in descriptions.items():
                value = unwrap(payload)
                if isinstance(value, dict):
                    rows.append({
                        "route": route,
                        "domain": value.get("domain") or value.get("domain_path"),
                        "description": value.get("description"),
                        "required_scopes": value.get("required_scopes") or value.get("scopes"),
                        "connectors": value.get("connectors") or value.get("providers"),
                    })
                else:
                    rows.append({"route": route, "domain": None, "description": None, "required_scopes": None, "connectors": None})
            contract = pd.DataFrame(rows)
            coverage = pd.Series({
                "domain_list_rows": len(as_rows(domains_list)),
                "domain_tree_rows": len(as_rows(domain_tree)),
                "aliases_rows": len(as_rows(aliases)),
                "described_routes": contract["description"].notna().sum() if not contract.empty else 0,
            })
            display(coverage)
            display(contract)
            """,
            """
            plot_data = pd.Series({
                "list": len(as_rows(domains_list)),
                "aliases": len(as_rows(aliases)),
                "descriptions": len(descriptions),
                "routes_checked": len(route_names),
            })
            plot_data.plot(kind="bar", title="Domain discovery surface")
            plt.ylabel("count")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "16_global_macro_sources.ipynb",
        "Global Macro Source Coverage",
        "Core data primitive",
        "Queries US, global and regional macro connectors to build a source coverage matrix for macro research.",
        [
            "qj.fred.get_cpi",
            "qj.fred.get_treasury_10y",
            "qj.imf.get_gdp_data",
            "qj.oecd.get_cpi_data",
            "qj.worldbank.get_indicator",
            "qj.dbnomics.get_inflation_rates",
            "qj.eurostat.get_eu_data",
        ],
        [
            """
            macro_calls = {
                "fred_cpi": safe_call("FRED CPI", qj.fred.get_cpi),
                "fred_10y": safe_call("FRED 10Y", qj.fred.get_treasury_10y),
                "imf_gdp": safe_call("IMF GDP", qj.imf.get_gdp_data, country="US"),
                "imf_inflation": safe_call("IMF inflation", qj.imf.get_inflation_data, country="US"),
                "oecd_cpi": safe_call("OECD CPI", qj.oecd.get_cpi_data, country="USA"),
                "worldbank_gdp": safe_call("World Bank GDP", qj.worldbank.get_indicator, country="US", indicator="NY.GDP.MKTP.CD"),
                "dbnomics_inflation": safe_call("DBnomics inflation", qj.dbnomics.get_inflation_rates, country="US"),
                "dbnomics_rates": safe_call("DBnomics interest rates", qj.dbnomics.get_interest_rates, country="US"),
                "eurostat": safe_call("Eurostat EU data", qj.eurostat.get_eu_data),
            }
            """,
            """
            coverage = []
            for name, payload in macro_calls.items():
                rows = as_rows(payload)
                value = unwrap(payload)
                coverage.append({
                    "source": name,
                    "available": payload is not None,
                    "rows": len(rows),
                    "shape": type(value).__name__,
                })
            coverage_df = pd.DataFrame(coverage).sort_values(["available", "rows"], ascending=False)
            display(coverage_df)
            """,
            """
            coverage_df.set_index("source")["rows"].plot(kind="bar", title="Macro connector row coverage")
            plt.ylabel("rows returned")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "17_options_vix_skew_term_structure.ipynb",
        "Options, VIX, SKEW and Term Structure",
        "Core data primitive",
        "Inspects volatility feeds and options context around VIX, VVIX, SKEW, term structure and option-chain metadata.",
        [
            "qj.cboe.get_vix_data",
            "qj.cboe.get_vvix_data",
            "qj.cboe.get_skew_index_data",
            "qj.cboe.get_vix_term_structure",
            "qj.cboe.get_options_expirations",
            "qj.cboe.get_options_chain",
        ],
        [
            """
            symbol = "SPY"
            vix_raw = safe_call("CBOE VIX", qj.cboe.get_vix_data, start_date="2020-01-01", end_date=END)
            vvix_raw = safe_call("CBOE VVIX", qj.cboe.get_vvix_data, start_date="2020-01-01", end_date=END)
            skew_raw = safe_call("CBOE SKEW", qj.cboe.get_skew_index_data, start_date="2020-01-01", end_date=END)
            term_raw = safe_call("CBOE VIX term structure", qj.cboe.get_vix_term_structure)
            expirations_raw = safe_call("CBOE options expirations", qj.cboe.get_options_expirations, symbol=symbol)
            chain_raw = safe_call("CBOE options chain", qj.cboe.get_options_chain, symbol=symbol)
            """,
            """
            def series_from_rows(payload: Any, name: str) -> pd.Series:
                rows = pd.DataFrame(as_rows(payload))
                if rows.empty:
                    return pd.Series(dtype=float, name=name)
                date_col = next((col for col in rows.columns if "date" in str(col).lower()), rows.columns[0])
                numeric_cols = rows.select_dtypes(include="number").columns.tolist()
                value_col = "close" if "close" in rows.columns else (numeric_cols[-1] if numeric_cols else None)
                if value_col is None:
                    return pd.Series(dtype=float, name=name)
                rows["date"] = pd.to_datetime(rows[date_col], errors="coerce")
                rows[name] = pd.to_numeric(rows[value_col], errors="coerce")
                return rows.dropna(subset=["date", name]).set_index("date")[name].sort_index()

            vol = pd.concat([
                series_from_rows(vix_raw, "vix"),
                series_from_rows(vvix_raw, "vvix"),
                series_from_rows(skew_raw, "skew"),
            ], axis=1).dropna(how="all")
            feed_summary = pd.Series({
                "vix_rows": len(as_rows(vix_raw)),
                "vvix_rows": len(as_rows(vvix_raw)),
                "skew_rows": len(as_rows(skew_raw)),
                "term_structure_rows": len(as_rows(term_raw)),
                "expiration_rows": len(as_rows(expirations_raw)),
                "chain_rows": len(as_rows(chain_raw)),
            })
            display(feed_summary)
            """,
            """
            if not vol.empty:
                vol.tail(1000).plot(title="Volatility feed context")
                plt.ylabel("index level")
                plt.show()
                latest = vol.tail(252).describe().T
                display(latest)
            """,
        ],
    ),
    NotebookSpec(
        "18_index_constituents_universe_build.ipynb",
        "Index Constituents and Universe Build",
        "Core data primitive",
        "Builds a research universe from index constituents, sector metadata, exchange symbols and price/liquidity filters.",
        [
            "qj.yf.get_sp500_stocks_info",
            "qj.yf.get_sp500_sectors",
            "qj.yf.get_sp500_index",
            "qj.eod.get_exchange_symbols",
            "qj.eod.get_historical_prices",
        ],
        [
            """
            sp500_info_raw = safe_call("YF S&P 500 stocks info", qj.yf.get_sp500_stocks_info)
            sectors_raw = safe_call("YF S&P 500 sectors", qj.yf.get_sp500_sectors)
            index_raw = safe_call("YF S&P 500 index", qj.yf.get_sp500_index)
            exchange_symbols_raw = safe_call("EOD exchange symbols", qj.eod.get_exchange_symbols, exchange="US")
            seed_symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "LLY", "AVGO"]
            prices, volumes = price_panel(seed_symbols, start="2023-01-01", end=END)
            """,
            """
            info = pd.DataFrame(as_rows(sp500_info_raw))
            sectors = pd.DataFrame(as_rows(sectors_raw))
            exchange_symbols = pd.DataFrame(as_rows(exchange_symbols_raw))
            ret = returns(prices)
            universe = pd.DataFrame(index=seed_symbols)
            universe["adv_usd_63d"] = dollar_adv(prices, volumes).iloc[-1].reindex(seed_symbols)
            universe["momentum_126d"] = prices.pct_change(126).iloc[-1].reindex(seed_symbols)
            universe["volatility_63d"] = ret.tail(63).std().reindex(seed_symbols) * np.sqrt(252)
            universe["liquid"] = universe["adv_usd_63d"] > 50_000_000
            universe["score"] = universe["momentum_126d"].rank(pct=True) - universe["volatility_63d"].rank(pct=True) * 0.35
            display(pd.Series({
                "sp500_info_rows": len(info),
                "sector_rows": len(sectors),
                "exchange_symbol_rows": len(exchange_symbols),
                "seed_symbols": len(seed_symbols),
            }))
            display(universe.sort_values("score", ascending=False))
            """,
            """
            universe[["adv_usd_63d", "momentum_126d", "volatility_63d"]].plot(kind="bar", subplots=True, layout=(1, 3), figsize=(15, 4), title="Universe build diagnostics")
            plt.tight_layout()
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "19_data_contract_lineage_audit.ipynb",
        "Data Contract, Lineage and Audit Pattern",
        "Core data primitive",
        "Shows how to retain provider, route, warnings, request metadata and source evidence next to normalized data outputs.",
        [
            "qj.eod.get_historical_prices",
            "qj.fmp.get_financial_ratios_ttm",
            "qj.sec.get_company_filings",
            "qj.openfigi.get_figi_data",
        ],
        [
            """
            symbol = "AAPL"
            calls = {
                "prices": safe_call("EOD historical prices", qj.eod.get_historical_prices, symbol=symbol, start_date="2024-01-01", end_date=END),
                "ratios": safe_call("FMP TTM ratios", qj.fmp.get_financial_ratios_ttm, symbol=symbol),
                "filings": safe_call("SEC company filings", qj.sec.get_company_filings, symbol=symbol, limit=10),
                "identity": safe_call("OpenFIGI identity", qj.openfigi.get_figi_data, symbol=symbol, exchange="US"),
            }
            """,
            """
            def audit_row(name: str, payload: Any, provider: str, route: str) -> dict[str, Any]:
                value = unwrap(payload)
                rows = as_rows(payload)
                meta = value.get("meta", {}) if isinstance(value, dict) and isinstance(value.get("meta"), dict) else {}
                return {
                    "dataset": name,
                    "provider": provider,
                    "route": route,
                    "available": payload is not None,
                    "rows": len(rows),
                    "shape": type(value).__name__,
                    "request_id": meta.get("request_id"),
                    "warnings": meta.get("warnings", []),
                }

            audit = pd.DataFrame([
                audit_row("prices", calls["prices"], "eod", "equity.pricing.get_historical_prices"),
                audit_row("ratios", calls["ratios"], "fmp", "equity.fundamentals.get_financial_ratios_ttm"),
                audit_row("filings", calls["filings"], "sec", "regulatory.sec.get_company_filings"),
                audit_row("identity", calls["identity"], "openfigi", "reference.identifiers.get_figi_data"),
            ])
            display(audit)
            """,
            """
            coverage = audit.set_index("dataset")["rows"]
            coverage.plot(kind="bar", title="Lineage packet coverage")
            plt.ylabel("rows returned")
            plt.show()
            lineage_packet = {
                "symbol": symbol,
                "datasets": audit.to_dict(orient="records"),
                "policy": {
                    "tenant_scoped": True,
                    "provider_secrets_server_side": True,
                    "retain_request_metadata": True,
                },
            }
            print(json.dumps(lineage_packet, indent=2, default=str)[:2000])
            """,
        ],
    ),
]


SPECS = [
    NotebookSpec(
        "22_hierarchical_risk_parity_hrp.ipynb",
        "Hierarchical Risk Parity vs Risk Parity vs Minimum Variance",
        "Portfolio construction",
        "Compares HRP-style recursive allocation, inverse-volatility risk parity and minimum variance using the same price panel.",
        ["qj.eod.get_historical_prices"],
        [
            """
            symbols = ["SPY", "TLT", "GLD", "DBC", "UUP"]
            prices, volumes = price_panel(symbols)
            ret = returns(prices).dropna()
            ret.tail()
            """,
            """
            def corr_sort_order(ret: pd.DataFrame) -> list[str]:
                corr = ret.corr().fillna(0)
                score = corr.mean().sort_values()
                return list(score.index)

            def cluster_var(cov: pd.DataFrame, assets: list[str]) -> float:
                sub = cov.loc[assets, assets]
                w = inverse_vol_weights(ret[assets])
                return float(w @ sub @ w)

            def hrp_weights(ret: pd.DataFrame) -> pd.Series:
                cov = ret.cov() * 252
                order = corr_sort_order(ret)
                weights = pd.Series(1.0, index=order)
                clusters = [order]
                while clusters:
                    cluster = clusters.pop(0)
                    if len(cluster) <= 1:
                        continue
                    split = len(cluster) // 2
                    left, right = cluster[:split], cluster[split:]
                    left_var = cluster_var(cov, left)
                    right_var = cluster_var(cov, right)
                    alpha = 1 - left_var / (left_var + right_var)
                    weights[left] *= alpha
                    weights[right] *= 1 - alpha
                    clusters.extend([left, right])
                return weights.reindex(ret.columns).fillna(0) / weights.sum()

            weights = pd.DataFrame({
                "HRP": hrp_weights(ret),
                "InverseVol": inverse_vol_weights(ret),
                "MinVariance": min_variance_weights(ret),
            })
            weights
            """,
            """
            portfolio = {col: portfolio_returns(ret, weights[col]) for col in weights.columns}
            summary = pd.DataFrame({name: performance_stats(series) for name, series in portfolio.items()}).T
            display(summary)
            display(weights.style.format("{:.2%}"))
            plot_nav(portfolio, "HRP vs inverse-vol vs min-variance")
            risk = pd.DataFrame({col: risk_contribution(ret, weights[col]) for col in weights.columns})
            risk.plot(kind="bar", title="Risk contribution by method")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "30_universe_construction_liquidity_screen.ipynb",
        "Universe Construction and Liquidity Screen",
        "Universe construction",
        "Builds an investable equity universe from prices, volumes, optional FMP screener output, TTM ratios and short-interest context.",
        ["qj.eod.get_historical_prices", "qj.fmp.get_stock_screener", "qj.fmp.get_financial_ratios_ttm", "qj.finra.get_short_interest"],
        [
            """
            seed_symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "JPM", "LLY", "XOM", "UNH", "V"]
            screener = safe_call("FMP screener", qj.fmp.get_stock_screener, marketCapMoreThan=10_000_000_000, limit=50)
            prices, volumes = price_panel(seed_symbols)
            ret = returns(prices)
            adv = dollar_adv(prices, volumes).iloc[-1]
            """,
            """
            ratio_rows = []
            for symbol in seed_symbols:
                payload = safe_call(f"FMP ratios {symbol}", qj.fmp.get_financial_ratios_ttm, symbol=symbol)
                data = unwrap(payload) or {}
                ratio_rows.append({
                    "symbol": symbol,
                    "pe_ttm": pd.to_numeric(data.get("peRatioTTM"), errors="coerce"),
                    "gross_margin_ttm": pd.to_numeric(data.get("grossProfitMarginTTM"), errors="coerce"),
                    "fcf_yield_ttm": 1 / pd.to_numeric(data.get("priceToFreeCashFlowRatioTTM"), errors="coerce")
                    if data.get("priceToFreeCashFlowRatioTTM") else np.nan,
                })
            ratios = pd.DataFrame(ratio_rows).set_index("symbol")
            short_interest = {
                symbol: safe_call(f"FINRA short interest {symbol}", qj.finra.get_short_interest, symbol=symbol)
                for symbol in seed_symbols[:5]
            }
            """,
            """
            features = pd.DataFrame(index=seed_symbols)
            features["adv_usd"] = adv.reindex(seed_symbols)
            features["volatility_63d"] = ret[seed_symbols].tail(63).std() * np.sqrt(252)
            features["momentum_126d"] = prices[seed_symbols].pct_change(126).iloc[-1]
            features = features.join(ratios)
            features["liquid"] = features["adv_usd"] > 50_000_000
            features["quality_score"] = features["gross_margin_ttm"].rank(pct=True) + features["fcf_yield_ttm"].rank(pct=True)
            features["risk_penalty"] = features["volatility_63d"].rank(pct=True)
            features["universe_score"] = features["quality_score"] + features["momentum_126d"].rank(pct=True) - features["risk_penalty"]
            universe = features.query("liquid").sort_values("universe_score", ascending=False)
            display(universe)
            universe[["adv_usd", "momentum_126d", "volatility_63d"]].plot(kind="bar", subplots=True, layout=(1, 3), figsize=(15, 4), title="Universe diagnostics")
            plt.tight_layout()
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "32_liquidity_capacity_impact.ipynb",
        "Liquidity Capacity and Market Impact",
        "Liquidity and capacity",
        "Estimates position capacity, participation limits and simple Amihud-style impact using adjusted prices and volume.",
        ["qj.eod.get_historical_prices", "qj.finra.get_short_interest"],
        [
            """
            symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "JPM"]
            prices, volumes = price_panel(symbols)
            ret = returns(prices)
            adv = dollar_adv(prices, volumes).iloc[-1]
            """,
            """
            dollar_volume = prices * volumes
            amihud = (ret.abs() / dollar_volume).rolling(63).mean().iloc[-1] * 1e9
            capacity = pd.DataFrame({
                "adv_usd": adv,
                "capacity_5pct_adv": adv * 0.05,
                "capacity_10pct_adv": adv * 0.10,
                "capacity_20pct_adv": adv * 0.20,
                "amihud_x1e9": amihud,
                "volatility_63d": ret.tail(63).std() * np.sqrt(252),
            }).sort_values("capacity_10pct_adv", ascending=False)
            display(capacity)
            """,
            """
            target_book = 50_000_000
            equal_position = target_book / len(symbols)
            capacity["target_position"] = equal_position
            capacity["days_to_trade_10pct_adv"] = capacity["target_position"] / capacity["capacity_10pct_adv"]
            capacity["capacity_flag"] = np.where(capacity["days_to_trade_10pct_adv"] > 5, "capacity constrained", "ok")
            display(capacity[["target_position", "days_to_trade_10pct_adv", "capacity_flag"]])
            capacity[["capacity_5pct_adv", "capacity_10pct_adv", "capacity_20pct_adv"]].plot(kind="bar", title="Capacity under participation limits")
            plt.ylabel("USD")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "33_stress_testing_macro_scenarios.ipynb",
        "Stress Testing Macro Scenarios",
        "Risk and scenarios",
        "Maps custom macro and market shocks onto a portfolio through observed factor betas.",
        ["qj.eod.get_historical_prices", "qj.fred.get_cpi", "qj.fred.get_effective_federal_funds_rate", "qj.cftc.get_cot_summary"],
        [
            """
            holdings = {"AAPL": 0.18, "MSFT": 0.18, "NVDA": 0.14, "GOOGL": 0.12, "AMZN": 0.12, "JPM": 0.10, "XOM": 0.08, "LLY": 0.08}
            factors = ["SPY", "TLT", "UUP", "GLD", "DBC"]
            prices, volumes = price_panel(list(holdings) + factors)
            ret = returns(prices)
            portfolio = portfolio_returns(ret, pd.Series(holdings))
            factor_ret = ret[factors]
            macro = {
                "cpi": safe_call("FRED CPI", qj.fred.get_cpi),
                "fed_funds": safe_call("FRED effective fed funds", qj.fred.get_effective_federal_funds_rate),
                "cot": safe_call("CFTC COT summary", qj.cftc.get_cot_summary, symbol="SPX"),
            }
            """,
            """
            betas = rolling_betas(portfolio, factor_ret, window=252).tail(1).T
            betas.columns = ["portfolio_beta"]
            scenarios = pd.DataFrame({
                "equity_selloff": {"SPY": -0.08, "TLT": 0.02, "UUP": 0.015, "GLD": 0.025, "DBC": -0.02},
                "rates_up": {"SPY": -0.03, "TLT": -0.07, "UUP": 0.02, "GLD": -0.01, "DBC": 0.01},
                "inflation_spike": {"SPY": -0.04, "TLT": -0.05, "UUP": 0.01, "GLD": 0.03, "DBC": 0.08},
                "risk_on": {"SPY": 0.05, "TLT": -0.01, "UUP": -0.01, "GLD": -0.01, "DBC": 0.02},
            }).T
            scenario_pnl = scenarios @ betas["portfolio_beta"]
            display(betas)
            display((scenario_pnl * 100).rename("estimated_portfolio_return_pct"))
            scenario_pnl.mul(100).plot(kind="bar", title="Scenario P&L proxy")
            plt.ylabel("%")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "34_index_replication_tracking_error.ipynb",
        "Index Replication and Tracking Error",
        "Portfolio construction",
        "Builds a constrained replication basket and measures tracking error against SPY.",
        ["qj.eod.get_historical_prices", "qj.yf.get_sp500_stocks_info"],
        [
            """
            benchmark = "SPY"
            candidates = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "JPM", "LLY", "XOM", "UNH", "V"]
            sp500_info = safe_call("S&P 500 constituents/info", qj.yf.get_sp500_stocks_info)
            prices, volumes = price_panel([benchmark] + candidates)
            ret = returns(prices).dropna()
            """,
            """
            y = ret[benchmark]
            x = ret[candidates]
            beta = np.linalg.lstsq(x.to_numpy(), y.to_numpy(), rcond=None)[0]
            weights = pd.Series(np.maximum(beta, 0), index=candidates)
            weights = weights / weights.sum()
            weights = weights.clip(upper=0.20)
            weights = weights / weights.sum()
            replica = portfolio_returns(ret, weights)
            active = replica - y.reindex(replica.index)
            tracking_error = active.std() * np.sqrt(252)
            info_ratio = active.mean() * 252 / tracking_error
            display(weights.sort_values(ascending=False).rename("replication_weight"))
            print(f"Tracking error: {tracking_error:.2%}; information ratio proxy: {info_ratio:.2f}")
            plot_nav({"replica": replica, "SPY": y}, "Index replication NAV")
            active.cumsum().plot(title="Cumulative active return")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "36_rolling_risk_budgeting_drawdown_control.ipynb",
        "Rolling Risk Budgeting and Drawdown Control",
        "Risk management",
        "Applies volatility targeting, rolling risk budgeting and drawdown exposure control to a multi-asset portfolio.",
        ["qj.eod.get_historical_prices", "qj.fred.get_effective_federal_funds_rate"],
        [
            """
            symbols = ["SPY", "TLT", "GLD", "DBC", "UUP"]
            prices, volumes = price_panel(symbols)
            ret = returns(prices).dropna()
            base_weights = inverse_vol_weights(ret)
            raw = portfolio_returns(ret, base_weights)
            """,
            """
            target_vol = 0.10
            realized_vol = raw.rolling(63).std() * np.sqrt(252)
            leverage = (target_vol / realized_vol).clip(0.25, 1.50).shift(1).fillna(1.0)
            nav_raw = (1 + raw).cumprod()
            drawdown = nav_raw / nav_raw.cummax() - 1
            dd_control = pd.Series(1.0, index=raw.index)
            dd_control[drawdown < -0.08] = 0.50
            dd_control[drawdown < -0.15] = 0.25
            managed = raw * leverage * dd_control
            summary = pd.DataFrame({"raw": performance_stats(raw), "managed": performance_stats(managed)}).T
            display(summary)
            plot_nav({"raw inverse-vol": raw, "vol-target + drawdown control": managed}, "Risk-budgeted NAV")
            pd.DataFrame({"leverage": leverage, "drawdown_control": dd_control}).plot(title="Risk controls")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "37_brinson_attribution_benchmark_relative.ipynb",
        "Brinson Benchmark-Relative Attribution",
        "Attribution",
        "Calculates allocation, selection and interaction effects using holdings, benchmark weights and sector return buckets.",
        ["qj.eod.get_historical_prices", "qj.yf.get_sp500_sectors"],
        [
            """
            sector_feed = safe_call("S&P 500 sectors", qj.yf.get_sp500_sectors)

            def normalize_sector_map(payload: Any) -> dict[str, str]:
                rows = as_rows(payload)
                sector_map = {}
                for item in rows:
                    symbol = item.get("symbol") or item.get("ticker") or item.get("Symbol")
                    sector = item.get("sector") or item.get("Sector") or item.get("gics_sector")
                    if symbol and sector:
                        sector_map[str(symbol).upper()] = str(sector)
                return sector_map

            sectors = normalize_sector_map(sector_feed)
            portfolio_w = pd.Series({"AAPL": 0.18, "MSFT": 0.18, "NVDA": 0.16, "GOOGL": 0.12, "AMZN": 0.12, "JPM": 0.10, "XOM": 0.06, "LLY": 0.08})
            benchmark_w = pd.Series({"AAPL": 0.12, "MSFT": 0.12, "NVDA": 0.10, "GOOGL": 0.09, "AMZN": 0.09, "JPM": 0.08, "XOM": 0.10, "LLY": 0.08})
            benchmark_w = benchmark_w / benchmark_w.sum()
            prices, volumes = price_panel(list(portfolio_w.index))
            period_return = prices.iloc[-1] / prices.iloc[0] - 1
            """,
            """
            df = pd.DataFrame({"sector": pd.Series(sectors), "portfolio_w": portfolio_w, "benchmark_w": benchmark_w, "return": period_return})
            df["sector"] = df["sector"].fillna("Unknown")
            sector = df.groupby("sector").apply(lambda x: pd.Series({
                "portfolio_w": x["portfolio_w"].sum(),
                "benchmark_w": x["benchmark_w"].sum(),
                "portfolio_return": np.average(x["return"], weights=x["portfolio_w"] / x["portfolio_w"].sum()),
                "benchmark_return": np.average(x["return"], weights=x["benchmark_w"] / x["benchmark_w"].sum()),
            }))
            benchmark_total = float((df["benchmark_w"] * df["return"]).sum())
            sector["allocation"] = (sector["portfolio_w"] - sector["benchmark_w"]) * (sector["benchmark_return"] - benchmark_total)
            sector["selection"] = sector["benchmark_w"] * (sector["portfolio_return"] - sector["benchmark_return"])
            sector["interaction"] = (sector["portfolio_w"] - sector["benchmark_w"]) * (sector["portfolio_return"] - sector["benchmark_return"])
            display(sector)
            sector[["allocation", "selection", "interaction"]].plot(kind="bar", stacked=True, title="Brinson attribution")
            plt.ylabel("Contribution")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "38_congress_smart_money_overlay.ipynb",
        "Congress Smart-Money Overlay",
        "Alternative data / signals",
        "Combines congressional trade feeds with price reactions to create a transparent event overlay.",
        ["qj.fmp.get_senate_trades", "qj.fmp.get_house_trades", "qj.eod.get_historical_prices"],
        [
            """
            symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
            senate = safe_call("FMP senate trades", qj.fmp.get_senate_trades, symbol="AAPL")
            house = safe_call("FMP house trades", qj.fmp.get_house_trades, symbol="AAPL")
            prices, volumes = price_panel(symbols)
            ret = returns(prices)
            """,
            """
            events = pd.concat([
                pd.DataFrame(as_rows(senate)).assign(source="senate"),
                pd.DataFrame(as_rows(house)).assign(source="house"),
            ], ignore_index=True)

            def first_present(columns: list[str], candidates: list[str]) -> str | None:
                normalized = {str(col).lower(): col for col in columns}
                for candidate in candidates:
                    if candidate.lower() in normalized:
                        return normalized[candidate.lower()]
                for col in columns:
                    low = str(col).lower()
                    if any(candidate.lower() in low for candidate in candidates):
                        return col
                return None

            if not events.empty:
                date_col = first_present(list(events.columns), ["transactionDate", "transaction_date", "disclosureDate", "reportingDate", "filingDate", "date"])
                symbol_col = first_present(list(events.columns), ["symbol", "ticker", "assetTicker", "asset_ticker"])
                events["event_date"] = pd.to_datetime(events[date_col], errors="coerce") if date_col else pd.NaT
                events["symbol"] = events[symbol_col].astype(str).str.upper() if symbol_col else "AAPL"
                event_returns = []
                for row in events.dropna(subset=["event_date"]).itertuples():
                    symbol = getattr(row, "symbol", "AAPL")
                    if symbol in prices:
                        idx = prices.index.searchsorted(row.event_date)
                        if 0 <= idx < len(prices) - 21:
                            event_returns.append({"symbol": symbol, "event_date": row.event_date, "fwd_21d": prices[symbol].iloc[idx + 21] / prices[symbol].iloc[idx] - 1})
                event_returns = pd.DataFrame(event_returns)
            else:
                event_returns = pd.DataFrame(columns=["symbol", "event_date", "fwd_21d"])
            display(events.head())
            display(event_returns.describe(include="all"))
            prices.div(prices.iloc[0]).plot(title="Price context for congress-trade overlay")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "39_institutional_crowding_13f_flows.ipynb",
        "Institutional Crowding and 13F Flow Signals",
        "Regulatory / crowding",
        "Uses SEC/FMP institutional ownership calls with returns data to build crowding, concentration and flow proxies.",
        ["qj.sec.get_institutional_holdings", "qj.fmp.get_institutional_holders", "qj.eod.get_historical_prices"],
        [
            """
            symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
            ownership = {
                symbol: safe_call(f"SEC institutional holdings {symbol}", qj.sec.get_institutional_holdings, symbol=symbol)
                for symbol in symbols[:3]
            }
            fmp_holders = {
                symbol: safe_call(f"FMP institutional holders {symbol}", qj.fmp.get_institutional_holders, symbol=symbol)
                for symbol in symbols[:3]
            }
            prices, volumes = price_panel(symbols)
            ret = returns(prices)
            """,
            """
            rows = []
            for symbol, payload in fmp_holders.items():
                holder_rows = as_rows(payload)
                values = []
                for item in holder_rows:
                    value = item.get("value") or item.get("marketValue") or item.get("shares")
                    values.append(pd.to_numeric(value, errors="coerce"))
                values = pd.Series(values).dropna()
                rows.append({
                    "symbol": symbol,
                    "holder_count": len(holder_rows),
                    "top10_concentration": values.nlargest(10).sum() / values.sum() if values.sum() else np.nan,
                })
            crowding = pd.DataFrame(rows).set_index("symbol")
            crowding["momentum_126d"] = prices.pct_change(126).iloc[-1].reindex(crowding.index)
            display(crowding)
            crowding[["holder_count", "top10_concentration", "momentum_126d"]].plot(kind="bar", subplots=True, layout=(1, 3), figsize=(15, 4), title="Crowding diagnostics")
            plt.tight_layout()
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "41_broad_event_study_pead.ipynb",
        "Broad Earnings Event Study and PEAD",
        "Event studies",
        "Extends earnings analysis into a multi-name post-earnings drift workflow with event windows and sector splits.",
        ["qj.fmp.get_earnings_calendar", "qj.fmp.get_earnings_surprises", "qj.eod.get_historical_prices"],
        [
            """
            symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
            calendar = safe_call("FMP earnings calendar", qj.fmp.get_earnings_calendar, from_date="2024-01-01", to_date=END)
            surprises = {symbol: safe_call(f"FMP earnings surprises {symbol}", qj.fmp.get_earnings_surprises, symbol=symbol) for symbol in symbols}
            prices, volumes = price_panel(symbols, start="2023-01-01", end=END)
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
                events = pd.DataFrame({"symbol": symbols, "event_date": [prices.index[-126]] * len(symbols), "surprise": np.nan})
            curves = []
            for row in events.dropna(subset=["event_date"]).itertuples():
                if row.symbol not in prices:
                    continue
                idx = prices.index.searchsorted(row.event_date)
                if idx < 21 or idx + 42 >= len(prices):
                    continue
                window = prices[row.symbol].iloc[idx - 10:idx + 43]
                curve = window / window.iloc[10] - 1
                curves.append(pd.Series(curve.values, index=range(-10, len(curve) - 10), name=row.symbol))
            event_curve = pd.concat(curves, axis=1) if curves else pd.DataFrame()
            display(events.head())
            if not event_curve.empty:
                event_curve.mean(axis=1).plot(title="Average post-earnings drift curve")
                plt.axvline(0, color="black", linestyle="--")
                plt.ylabel("Return vs event day")
                plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "42_factor_risk_model_construction.ipynb",
        "Factor Risk Model Construction",
        "Risk models",
        "Builds sample, shrinkage and factor-model covariance estimates from price and factor returns.",
        ["qj.eod.get_historical_prices", "qj.ff.get_factors"],
        [
            """
            assets = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
            factor_symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
            ff_factors = safe_call("Fama-French factors", qj.ff.get_factors, region="US")
            prices, volumes = price_panel(assets + factor_symbols)
            ret = returns(prices).dropna()
            asset_ret = ret[assets]
            factor_ret = ret[factor_symbols]
            """,
            """
            sample_cov = asset_ret.cov() * 252
            diagonal = pd.DataFrame(np.diag(np.diag(sample_cov)), index=assets, columns=assets)
            shrinkage_cov = 0.75 * sample_cov + 0.25 * diagonal
            betas = pd.DataFrame({asset: rolling_betas(asset_ret[asset], factor_ret, window=252).iloc[-1] for asset in assets}).T
            factor_cov = factor_ret.cov() * 252
            systematic = betas @ factor_cov @ betas.T
            resid = asset_ret - factor_ret @ betas.T
            specific = pd.DataFrame(np.diag(resid.var() * 252), index=assets, columns=assets)
            factor_model_cov = systematic + specific
            display(betas)
            display(pd.DataFrame({
                "sample_vol": np.sqrt(np.diag(sample_cov)),
                "shrinkage_vol": np.sqrt(np.diag(shrinkage_cov)),
                "factor_model_vol": np.sqrt(np.diag(factor_model_cov)),
            }, index=assets))
            plt.imshow(factor_model_cov.corr(), cmap="RdBu", vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(assets)), assets, rotation=45)
            plt.yticks(range(len(assets)), assets)
            plt.title("Factor-model covariance correlation")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "44_cta_futures_carry_trend_macro.ipynb",
        "CTA Futures Carry, Trend and Macro Overlay",
        "CTA / futures",
        "Creates a simple CTA-like signal book using trend proxies, optional COT positioning and macro context.",
        ["qj.eod.get_historical_prices", "qj.eod.get_futures_pricing", "qj.cftc.get_cot_summary", "qj.fred.get_effective_federal_funds_rate"],
        [
            """
            proxies = ["DBC", "GLD", "TLT", "UUP", "SPY"]
            cot = safe_call("CFTC COT summary", qj.cftc.get_cot_summary, symbol="GC")
            futures = safe_call("EOD futures pricing", qj.eod.get_futures_pricing, symbol="CL")
            fed = safe_call("FRED fed funds", qj.fred.get_effective_federal_funds_rate)
            prices, volumes = price_panel(proxies)
            ret = returns(prices)
            """,
            """
            trend_63 = prices.pct_change(63)
            trend_252 = prices.pct_change(252)
            vol = ret.rolling(63).std() * np.sqrt(252)
            signal = (0.5 * np.sign(trend_63) + 0.5 * np.sign(trend_252)).shift(1).fillna(0)
            scaled = signal.div(vol).replace([np.inf, -np.inf], np.nan).fillna(0)
            weights = scaled.div(scaled.abs().sum(axis=1), axis=0).fillna(0)
            cta_ret = (weights.shift(1) * ret).sum(axis=1)
            display(weights.tail())
            plot_nav({"CTA trend proxy": cta_ret, "SPY": ret["SPY"]}, "CTA trend proxy vs SPY")
            weights.tail(252).plot(title="CTA proxy weights")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "45_holdings_based_vs_returns_based.ipynb",
        "Holdings-Based vs Returns-Based Analysis",
        "Attribution / ownership",
        "Compares 13F/holdings-style concentration with returns-based factor exposure analysis.",
        ["qj.sec.get_institutional_portfolio_summary", "qj.fmp.get_institutional_portfolio", "qj.eod.get_historical_prices"],
        [
            """
            portfolio_symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM"]
            factors = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
            sec_summary = safe_call("SEC institutional portfolio summary", qj.sec.get_institutional_portfolio_summary, cik="0001067983")
            fmp_portfolio = safe_call("FMP institutional portfolio", qj.fmp.get_institutional_portfolio, cik="0001067983")
            prices, volumes = price_panel(portfolio_symbols + factors)
            ret = returns(prices).dropna()
            weights = pd.Series(1 / len(portfolio_symbols), index=portfolio_symbols)
            port_ret = portfolio_returns(ret, weights)
            """,
            """
            betas = rolling_betas(port_ret, ret[factors], window=252)
            concentration = pd.Series({
                "top_1_weight": weights.max(),
                "top_3_weight": weights.nlargest(3).sum(),
                "hhi": float((weights ** 2).sum()),
                "effective_names": float(1 / (weights ** 2).sum()),
            })
            display(concentration)
            display(betas.tail())
            betas.plot(title="Returns-based rolling exposure")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "46_options_overlay_strategies.ipynb",
        "Options Overlay Strategies",
        "Options / overlays",
        "Uses option-chain, VIX/SKEW and underlying returns to compare covered-call, put-write and collar payoff profiles.",
        ["qj.cboe.get_options_expirations", "qj.cboe.get_options_chain", "qj.cboe.get_vix_data", "qj.cboe.get_skew_index_data", "qj.eod.get_historical_prices"],
        [
            """
            symbol = "SPY"
            expirations = safe_call("CBOE expirations", qj.cboe.get_options_expirations, symbol=symbol)
            chain = safe_call("CBOE options chain", qj.cboe.get_options_chain, symbol=symbol)
            vix = safe_call("CBOE VIX", qj.cboe.get_vix_data, start_date=START, end_date=END)
            skew = safe_call("CBOE SKEW", qj.cboe.get_skew_index_data, start_date=START, end_date=END)
            prices, volumes = price_panel([symbol])
            ret = returns(prices)[symbol]
            """,
            """
            monthly = ret.resample("M").sum()
            covered_call = monthly.clip(upper=0.03) + 0.006
            put_write = monthly.clip(lower=-0.05) + 0.004
            collar = monthly.clip(lower=-0.04, upper=0.025) + 0.001
            overlays = pd.DataFrame({"underlying": monthly, "covered_call": covered_call, "put_write": put_write, "collar": collar}).dropna()
            display(overlays.apply(performance_stats).T)
            plot_nav({col: overlays[col] for col in overlays.columns}, "Options overlay payoff proxies")
            overlays.cumsum().plot(title="Cumulative monthly overlay returns")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "47_tactical_asset_allocation_macro_valuation.ipynb",
        "Tactical Asset Allocation with Macro and Valuation",
        "Asset allocation",
        "Combines macro, valuation and sentiment feeds with market data to create a dynamic allocation rule.",
        ["qj.fred.get_cpi", "qj.fred.get_effective_federal_funds_rate", "qj.multpl.get_shiller_pe_ratio", "qj.cnnf.get_latest_fear_greed_value", "qj.eod.get_historical_prices"],
        [
            """
            assets = ["SPY", "TLT", "GLD", "DBC", "UUP"]
            macro = {
                "cpi": safe_call("FRED CPI", qj.fred.get_cpi),
                "fed_funds": safe_call("FRED fed funds", qj.fred.get_effective_federal_funds_rate),
                "shiller_pe": safe_call("Multpl Shiller PE", qj.multpl.get_shiller_pe_ratio),
                "fear_greed": safe_call("CNN fear & greed", qj.cnnf.get_latest_fear_greed_value),
            }
            prices, volumes = price_panel(assets)
            ret = returns(prices)
            """,
            """
            momentum = prices.pct_change(126)
            vol = ret.rolling(63).std() * np.sqrt(252)
            raw_score = momentum.rank(axis=1, pct=True) - vol.rank(axis=1, pct=True) * 0.35
            weights = raw_score.clip(lower=0).div(raw_score.clip(lower=0).sum(axis=1), axis=0).fillna(0)
            weights = weights.rolling(21).mean().fillna(method="bfill")
            taa = (weights.shift(1) * ret).sum(axis=1)
            equal = ret.mean(axis=1)
            display(weights.tail())
            plot_nav({"macro/valuation TAA proxy": taa, "equal-weight": equal}, "Tactical allocation proxy")
            weights.tail(504).plot(title="Allocation weights")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "48_end_to_end_research_to_book.ipynb",
        "End-to-End Research to Book",
        "End-to-end workflow",
        "Runs universe scoring, weight construction, risk attribution, drawdown review and report-ready outputs in one notebook.",
        ["qj.eod.get_historical_prices", "qj.fmp.get_financial_ratios_ttm", "qj.finra.get_short_interest", "qj.insider.latest_trades"],
        [
            """
            universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "JPM", "LLY", "XOM"]
            prices, volumes = price_panel(universe)
            ret = returns(prices)
            insider = safe_call("Insider latest trades", qj.insider.latest_trades, symbols=universe[:5])
            """,
            """
            ratios = []
            for symbol in universe:
                data = unwrap(safe_call(f"FMP ratios {symbol}", qj.fmp.get_financial_ratios_ttm, symbol=symbol)) or {}
                ratios.append({"symbol": symbol, "gross_margin": pd.to_numeric(data.get("grossProfitMarginTTM"), errors="coerce"), "pe": pd.to_numeric(data.get("peRatioTTM"), errors="coerce")})
            ratios = pd.DataFrame(ratios).set_index("symbol")
            score = (
                prices.pct_change(126).iloc[-1].rank(pct=True)
                - ret.tail(63).std().rank(pct=True)
                + ratios["gross_margin"].rank(pct=True).reindex(universe).fillna(0)
                - ratios["pe"].rank(pct=True).reindex(universe).fillna(0) * 0.35
            )
            raw = score.clip(lower=0)
            weights = raw / raw.sum()
            book_ret = portfolio_returns(ret, weights)
            report = pd.DataFrame({
                "weight": weights,
                "risk_contribution": risk_contribution(ret, weights),
                "momentum_126d": prices.pct_change(126).iloc[-1],
                "vol_63d": ret.tail(63).std() * np.sqrt(252),
            }).sort_values("weight", ascending=False)
            display(report)
            display(performance_stats(book_ret))
            plot_nav({"research book": book_ret, "equal universe": ret[universe].mean(axis=1)}, "Research-to-book NAV")
            report[["weight", "risk_contribution"]].plot(kind="bar", title="Weight vs risk contribution")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "50_integrated_daily_risk_attribution_report.ipynb",
        "Integrated Daily Risk Attribution Report",
        "Daily PM report",
        "Produces a daily intelligence packet from holdings, factor moves, earnings/filings/insider context and risk contributors.",
        ["qj.eod.get_historical_prices", "qj.sec.get_recent_filings", "qj.insider.latest_trades", "qj.fmp.get_earnings_calendar"],
        [
            """
            holdings = pd.Series({"AAPL": 0.18, "MSFT": 0.18, "NVDA": 0.16, "GOOGL": 0.12, "AMZN": 0.12, "JPM": 0.10, "XOM": 0.06, "LLY": 0.08})
            factors = ["SPY", "QQQ", "TLT", "GLD", "UUP"]
            prices, volumes = price_panel(list(holdings.index) + factors)
            ret = returns(prices)
            filings = safe_call("SEC recent filings", qj.sec.get_recent_filings, limit=20)
            insider = safe_call("Insider latest trades", qj.insider.latest_trades, symbols=list(holdings.index))
            earnings = safe_call("FMP earnings calendar", qj.fmp.get_earnings_calendar, from_date=END, to_date=END)
            """,
            """
            latest_returns = ret[list(holdings.index)].iloc[-1]
            contribution = holdings * latest_returns
            book_ret = portfolio_returns(ret, holdings)
            betas = rolling_betas(book_ret, ret[factors], window=126).tail(1).T
            risk = risk_contribution(ret[list(holdings.index)], holdings)
            report = pd.DataFrame({
                "weight": holdings,
                "latest_return": latest_returns,
                "daily_contribution": contribution,
                "risk_contribution": risk,
            }).sort_values("daily_contribution")
            display(report)
            display(betas.rename(columns={betas.columns[0]: "latest_beta"}) if not betas.empty else betas)
            report[["daily_contribution", "risk_contribution"]].plot(kind="bar", title="Daily contribution and risk")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "52_turnover_cost_drag.ipynb",
        "Turnover and Cost Drag",
        "Execution-aware backtesting",
        "Quantifies how turnover, slippage assumptions and rebalance frequency change an otherwise attractive signal.",
        ["qj.eod.get_historical_prices"],
        [
            """
            symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "JPM"]
            prices, volumes = price_panel(symbols)
            ret = returns(prices)
            """,
            """
            def momentum_weights(prices: pd.DataFrame, lookback: int = 126, top_n: int = 4) -> pd.DataFrame:
                signal = prices.pct_change(lookback)
                weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
                for dt, row in signal.iterrows():
                    winners = row.nlargest(top_n).dropna().index
                    if len(winners):
                        weights.loc[dt, winners] = 1 / len(winners)
                return weights

            weights = momentum_weights(prices).resample("M").last().reindex(prices.index).ffill().fillna(0)
            gross = (weights.shift(1) * ret).sum(axis=1)
            turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
            results = {}
            for bps in [0, 5, 10, 25, 50]:
                net = gross - turnover * (bps / 10000)
                results[f"{bps} bps"] = net
            summary = pd.DataFrame({name: performance_stats(series) for name, series in results.items()}).T
            display(summary)
            plot_nav(results, "Turnover cost drag")
            turnover.rolling(21).sum().plot(title="Rolling turnover")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "53_backtest_with_risk_management_full.ipynb",
        "Backtest with Risk Management",
        "Backtesting",
        "Compares a raw momentum strategy with a risk-managed variant using volatility targeting, drawdown controls and concentration limits.",
        ["qj.eod.get_historical_prices", "qj.bt.prepare", "qj.analytics.hv"],
        [
            """
            symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
            bt_prepare = safe_call("BT prepare metadata", qj.bt.prepare, symbols=symbols, start=START, end=END)
            hv = safe_call("Analytics historical volatility", qj.analytics.hv, symbol="SPY", start=START, end=END)
            prices, volumes = price_panel(symbols)
            ret = returns(prices)
            """,
            """
            signal = prices.pct_change(126).rank(axis=1, pct=True)
            raw_weights = signal.clip(lower=0).div(signal.clip(lower=0).sum(axis=1), axis=0).fillna(0)
            raw_weights = raw_weights.clip(upper=0.35)
            raw_weights = raw_weights.div(raw_weights.sum(axis=1), axis=0).fillna(0)
            raw_ret = (raw_weights.shift(1) * ret).sum(axis=1)
            vol = raw_ret.rolling(63).std() * np.sqrt(252)
            scale = (0.12 / vol).clip(0.25, 1.25).shift(1).fillna(1.0)
            nav = (1 + raw_ret).cumprod()
            dd = nav / nav.cummax() - 1
            managed_ret = raw_ret * scale * np.where(dd < -0.10, 0.5, 1.0)
            summary = pd.DataFrame({"raw": performance_stats(raw_ret), "risk_managed": performance_stats(managed_ret)}).T
            display(summary)
            plot_nav({"raw momentum": raw_ret, "risk-managed": managed_ret}, "Backtest with risk management")
            pd.DataFrame({"scale": scale, "drawdown": dd}).plot(subplots=True, title="Risk controls")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "55_factor_timing_dynamic_exposures.ipynb",
        "Factor Timing and Dynamic Exposures",
        "Factor timing",
        "Uses macro regimes and factor proxies to time equity, growth, small-cap, duration and gold exposures.",
        ["qj.eod.get_historical_prices", "qj.ff.get_factors", "qj.fred.get_cpi", "qj.fred.get_effective_federal_funds_rate"],
        [
            """
            factors = ["SPY", "QQQ", "IWM", "TLT", "GLD", "DBC"]
            ff = safe_call("Fama-French factors", qj.ff.get_factors, region="US")
            cpi = safe_call("FRED CPI", qj.fred.get_cpi)
            fed = safe_call("FRED fed funds", qj.fred.get_effective_federal_funds_rate)
            prices, volumes = price_panel(factors)
            ret = returns(prices)
            """,
            """
            momentum = prices.pct_change(126)
            trend = zscore(momentum.mean(axis=1), 252)
            inflation_proxy = ret["DBC"].rolling(63).sum() - ret["TLT"].rolling(63).sum()
            regime_risk_on = (trend > 0).astype(float)
            weights = pd.DataFrame(index=ret.index, columns=factors, dtype=float)
            weights["SPY"] = 0.25 + 0.20 * regime_risk_on
            weights["QQQ"] = 0.20 + 0.15 * regime_risk_on
            weights["IWM"] = 0.15 * regime_risk_on
            weights["TLT"] = 0.25 - 0.10 * regime_risk_on - 0.10 * (inflation_proxy > 0).astype(float)
            weights["GLD"] = 0.15 + 0.10 * (inflation_proxy > 0).astype(float)
            weights = weights.clip(lower=0).div(weights.sum(axis=1), axis=0).fillna(method="bfill")
            dynamic = (weights.shift(1) * ret).sum(axis=1)
            equal = ret.mean(axis=1)
            display(weights.tail())
            plot_nav({"dynamic factor timing": dynamic, "equal factors": equal}, "Dynamic factor timing")
            weights.tail(504).plot(title="Dynamic exposures")
            plt.show()
            """,
        ],
    ),
    NotebookSpec(
        "57_crypto_perps_basis_funding_arbitrage.ipynb",
        "Crypto Perps Basis and Funding Arbitrage",
        "Crypto / derivatives",
        "Pulls CCXT-style spot/perp data and funding history where available, then builds funding and basis diagnostics.",
        ["qj.ccxt.get_historical_prices", "qj.ccxt.get_historical_funding_rates", "qj.ccxt.get_open_interest", "qj.coingecko.get_historical_prices"],
        [
            """
            btc_spot = safe_call("CCXT BTC spot OHLCV", qj.ccxt.get_historical_prices, symbol="BTC/USDT", exchange="binance", timeframe="1d", since=START)
            funding = safe_call("CCXT BTC funding", qj.ccxt.get_historical_funding_rates, symbol="BTC/USDT", exchange="binance")
            open_interest = safe_call("CCXT BTC open interest", qj.ccxt.get_open_interest, symbol="BTC/USDT", exchange="binance")
            gecko = safe_call("CoinGecko BTC historical", qj.coingecko.get_historical_prices, coin_id="bitcoin", vs_currency="usd", days="max")
            """,
            """
            rows = as_rows(btc_spot) or as_rows(gecko)
            prices_df = pd.DataFrame(rows)
            if not prices_df.empty:
                date_col = "date" if "date" in prices_df else prices_df.columns[0]
                numeric_cols = prices_df.select_dtypes(include="number").columns
                value_col = "close" if "close" in prices_df else (numeric_cols[-1] if len(numeric_cols) else prices_df.columns[-1])
                prices_df["date"] = pd.to_datetime(prices_df[date_col], errors="coerce")
                prices_df["price"] = pd.to_numeric(prices_df[value_col], errors="coerce")
                prices_df = prices_df.dropna(subset=["date", "price"]).set_index("date").sort_index()
                btc_ret = prices_df["price"].pct_change().dropna()
            else:
                btc_ret = pd.Series(dtype=float)
            funding_df = pd.DataFrame(as_rows(funding))
            display(funding_df.head())
            if not btc_ret.empty:
                nav = (1 + btc_ret).cumprod()
                nav.plot(title="BTC spot context for funding/basis research")
                plt.show()
                display(performance_stats(btc_ret))
            """,
        ],
    ),
]


ADVANCED_FULL_SPECS = [
    NotebookSpec(
        "61_vectorized_strategy_grid.ipynb",
        "Vectorized Strategy Grid",
        "Advanced buy-side diagnostics",
        "Runs a complete SMA parameter sweep from live QuantJourney price data, including transaction costs and robustness surfaces.",
        ["qj.eod.get_historical_prices"],
        [
            """
            prices, volumes = price_panel(["SPY"])
            spy = prices["SPY"].dropna()
            ret = spy.pct_change().fillna(0)
            fast_windows = [5, 10, 15, 20, 30, 40, 50]
            slow_windows = [60, 80, 100, 125, 150, 200]
            """,
            """
            def sma_strategy(price: pd.Series, fast: int, slow: int, cost_bps: float = 5.0) -> pd.Series:
                fast_ma = price.rolling(fast).mean()
                slow_ma = price.rolling(slow).mean()
                position = (fast_ma > slow_ma).astype(float).shift(1).fillna(0)
                turnover = position.diff().abs().fillna(position.abs())
                return position * price.pct_change().fillna(0) - turnover * cost_bps / 10000

            sharpe = pd.DataFrame(index=fast_windows, columns=slow_windows, dtype=float)
            total_return = sharpe.copy()
            max_dd = sharpe.copy()
            strategies = {}
            for fast in fast_windows:
                for slow in slow_windows:
                    if fast >= slow:
                        continue
                    sret = sma_strategy(spy, fast, slow)
                    strategies[(fast, slow)] = sret
                    st = performance_stats(sret)
                    sharpe.loc[fast, slow] = st["sharpe"]
                    total_return.loc[fast, slow] = st["total_return"]
                    max_dd.loc[fast, slow] = st["max_drawdown"]
            best = sharpe.stack().idxmax()
            print("Best fast/slow:", best)
            display(sharpe)
            """,
            """
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            for ax, data, title in zip(axes, [sharpe, total_return, max_dd], ["Sharpe", "Total return", "Max drawdown"]):
                im = ax.imshow(data.astype(float), aspect="auto")
                ax.set_xticks(range(len(data.columns)), data.columns)
                ax.set_yticks(range(len(data.index)), data.index)
                ax.set_title(title)
                fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.show()
            plot_nav({f"SMA {best[0]}/{best[1]}": strategies[best], "SPY": ret}, "Best grid strategy vs SPY")
            """,
        ],
        mirror_to_advanced=True,
    ),
    NotebookSpec(
        "62_walk_forward_robustness.ipynb",
        "Walk-Forward Robustness",
        "Advanced buy-side diagnostics",
        "Selects parameters on rolling training windows and evaluates the next out-of-sample fold.",
        ["qj.eod.get_historical_prices"],
        [
            """
            prices, volumes = price_panel(["SPY"])
            spy = prices["SPY"].dropna()
            fast_windows = [5, 10, 15, 20, 30, 40, 50]
            slow_windows = [60, 80, 100, 125, 150, 200]

            def sma_strategy(price: pd.Series, fast: int, slow: int, cost_bps: float = 5.0) -> pd.Series:
                fast_ma = price.rolling(fast).mean()
                slow_ma = price.rolling(slow).mean()
                position = (fast_ma > slow_ma).astype(float).shift(1).fillna(0)
                turnover = position.diff().abs().fillna(position.abs())
                return position * price.pct_change().fillna(0) - turnover * cost_bps / 10000
            """,
            """
            strategy_map = {(f, s): sma_strategy(spy, f, s) for f in fast_windows for s in slow_windows if f < s}
            train_days = 504
            test_days = 126
            records = []
            oos = []
            for fold, start_idx in enumerate(range(train_days, len(spy) - test_days, test_days), start=1):
                train_idx = spy.index[start_idx - train_days:start_idx]
                test_idx = spy.index[start_idx:start_idx + test_days]
                train_scores = {k: performance_stats(v.reindex(train_idx).fillna(0))["sharpe"] for k, v in strategy_map.items()}
                best = max(train_scores, key=lambda k: -np.inf if pd.isna(train_scores[k]) else train_scores[k])
                test_ret = strategy_map[best].reindex(test_idx).fillna(0)
                oos.append(test_ret)
                records.append({"fold": fold, "best_fast": best[0], "best_slow": best[1], "train_sharpe": train_scores[best], "test_sharpe": performance_stats(test_ret)["sharpe"], "test_return": (1 + test_ret).prod() - 1})
            wf = pd.DataFrame(records)
            oos_ret = pd.concat(oos).sort_index()
            display(wf)
            """,
            """
            wf[["train_sharpe", "test_sharpe"]].plot(kind="bar", title="Walk-forward train/test Sharpe")
            plt.show()
            plot_nav({"walk-forward": oos_ret, "SPY": spy.pct_change().reindex(oos_ret.index).fillna(0)}, "Walk-forward out-of-sample NAV")
            """,
        ],
        mirror_to_advanced=True,
    ),
    NotebookSpec(
        "63_monte_carlo_tail_risk.ipynb",
        "Monte Carlo Tail Risk",
        "Advanced buy-side diagnostics",
        "Bootstraps portfolio returns into one-year fan charts and terminal-return tail distributions.",
        ["qj.eod.get_historical_prices"],
        [
            """
            symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
            prices, volumes = price_panel(symbols)
            ret = returns(prices).dropna()
            weights = pd.Series(1 / len(symbols), index=symbols)
            port = portfolio_returns(ret, weights)
            """,
            """
            rng = np.random.default_rng(42)
            horizon = 252
            paths = 1000
            samples = rng.choice(port.dropna().to_numpy(), size=(horizon, paths), replace=True)
            sim = np.cumprod(1 + samples, axis=0)
            bands = pd.DataFrame(np.percentile(sim, [5, 25, 50, 75, 95], axis=1).T, columns=["p05", "p25", "p50", "p75", "p95"])
            terminal = pd.Series(sim[-1] - 1)
            display(terminal.describe(percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]))
            """,
            """
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes[0].fill_between(bands.index, bands["p05"], bands["p95"], alpha=0.15, label="5-95%")
            axes[0].fill_between(bands.index, bands["p25"], bands["p75"], alpha=0.25, label="25-75%")
            axes[0].plot(bands["p50"], label="median")
            axes[0].set_title("Monte Carlo fan")
            axes[0].legend()
            terminal.mul(100).hist(ax=axes[1], bins=40)
            axes[1].set_title("Terminal return distribution")
            plt.show()
            """,
        ],
        mirror_to_advanced=True,
    ),
    NotebookSpec(
        "64_correlation_regime_lab.ipynb",
        "Correlation Regime Lab",
        "Advanced buy-side diagnostics",
        "Calculates recent cross-asset correlation, rolling pairwise regimes and drawdown context.",
        ["qj.eod.get_historical_prices"],
        [
            """
            symbols = ["SPY", "TLT", "GLD", "DBC", "UUP"]
            prices, volumes = price_panel(symbols)
            ret = returns(prices).dropna()
            recent_corr = ret.tail(252).corr()
            rolling_avg = ret.rolling(63).corr().groupby(level=0).apply(lambda x: x.where(np.triu(np.ones(x.shape), 1).astype(bool)).stack().mean())
            spy_nav = (1 + ret["SPY"]).cumprod()
            spy_dd = spy_nav / spy_nav.cummax() - 1
            display(recent_corr)
            """,
            """
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            im = axes[0].imshow(recent_corr, vmin=-1, vmax=1, cmap="RdBu")
            axes[0].set_xticks(range(len(symbols)), symbols, rotation=45)
            axes[0].set_yticks(range(len(symbols)), symbols)
            axes[0].set_title("Recent correlation")
            fig.colorbar(im, ax=axes[0])
            rolling_avg.plot(ax=axes[1], title="Rolling average pairwise correlation")
            spy_dd.mul(100).plot(ax=axes[2], title="SPY drawdown context")
            plt.tight_layout()
            plt.show()
            """,
        ],
        mirror_to_advanced=True,
    ),
    NotebookSpec(
        "65_drawdown_diagnostics.ipynb",
        "Drawdown Diagnostics",
        "Advanced buy-side diagnostics",
        "Computes SMA 5/125 equity, underwater drawdown, rolling volatility and exposure state from live SPY prices.",
        ["qj.eod.get_historical_prices"],
        [
            """
            prices, volumes = price_panel(["SPY"])
            spy = prices["SPY"].dropna()
            fast, slow = 5, 125
            fast_ma = spy.rolling(fast).mean()
            slow_ma = spy.rolling(slow).mean()
            position = (fast_ma > slow_ma).astype(float).shift(1).fillna(0)
            turnover = position.diff().abs().fillna(position.abs())
            strategy_ret = position * spy.pct_change().fillna(0) - turnover * 0.0005
            buy_hold = spy.pct_change().fillna(0)
            """,
            """
            strategy_nav = (1 + strategy_ret).cumprod()
            benchmark_nav = (1 + buy_hold).cumprod()
            drawdown = strategy_nav / strategy_nav.cummax() - 1
            rolling_vol = strategy_ret.rolling(63).std() * np.sqrt(252)
            display(pd.DataFrame({"strategy": performance_stats(strategy_ret), "buy_hold": performance_stats(buy_hold)}).T)
            """,
            """
            fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
            strategy_nav.plot(ax=axes[0], label="SMA 5/125")
            benchmark_nav.plot(ax=axes[0], label="buy & hold")
            axes[0].legend()
            axes[0].set_title("Strategy equity vs benchmark")
            drawdown.mul(100).plot(ax=axes[1], title="Underwater drawdown")
            rolling_vol.mul(100).plot(ax=axes[2], title="Rolling 63D volatility")
            position.plot(ax=axes[3], title="Exposure state")
            plt.tight_layout()
            plt.show()
            """,
        ],
        mirror_to_advanced=True,
    ),
    NotebookSpec(
        "66_factor_exposure_diagnostics.ipynb",
        "Factor Exposure Diagnostics",
        "Advanced buy-side diagnostics",
        "Estimates rolling 126-day factor betas and recent contribution proxy for a mega-cap equity basket.",
        ["qj.eod.get_historical_prices"],
        [
            """
            holdings = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
            factors = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
            prices, volumes = price_panel(holdings + factors)
            ret = returns(prices).dropna()
            weights = pd.Series(1 / len(holdings), index=holdings)
            port = portfolio_returns(ret, weights)
            betas = rolling_betas(port, ret[factors], window=126)
            """,
            """
            latest_beta = betas.iloc[-1] if not betas.empty else pd.Series(dtype=float)
            factor_quarter = ret[factors].tail(63).sum()
            contribution_proxy = latest_beta * factor_quarter
            display(latest_beta.rename("latest_beta"))
            display(contribution_proxy.rename("quarter_contribution_proxy"))
            """,
            """
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            betas.plot(ax=axes[0], title="Rolling 126D betas")
            latest_beta.plot(kind="bar", ax=axes[1], title="Latest exposure")
            contribution_proxy.mul(100).plot(kind="bar", ax=axes[2], title="Quarter contribution proxy")
            plt.tight_layout()
            plt.show()
            """,
        ],
        mirror_to_advanced=True,
    ),
]


from multisource_candidate_specs import get_multisource_workflow_specs


MULTI_SOURCE_WORKFLOW_SPECS = get_multisource_workflow_specs(NotebookSpec)
ALL_GENERATED_SPECS = [*CORE_PRIMITIVE_SPECS, *SPECS, *ADVANCED_FULL_SPECS, *MULTI_SOURCE_WORKFLOW_SPECS]


def clean_example_notebooks() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    for path in CANDIDATES.glob("*.ipynb"):
        if path.name not in PRESERVED_SOURCE_CANDIDATE_NAMES:
            path.unlink()
    index = CANDIDATES / "INDEX.md"
    if index.exists():
        index.unlink()


def copy_existing() -> list[tuple[str, str, str]]:
    copied: list[tuple[str, str, str]] = []
    for source_dir, names, category, note in [
        (ROOT / "notebooks" / "core", EXISTING_CORE, "Existing core notebook", "copied from notebooks/core"),
        (ROOT / "notebooks" / "buy_side", EXISTING_BUY_SIDE, "Existing buy-side notebook", "copied from notebooks/buy_side"),
    ]:
        for name in names:
            src = source_dir / name
            dst = CANDIDATES / name
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, dst)
            copied.append((name, category, note))
    return copied


def write_generated(specs: Iterable[NotebookSpec]) -> list[str]:
    written: list[str] = []
    for spec in specs:
        nb = notebook(spec)
        out = CANDIDATES / spec.filename
        out.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
        written.append(spec.filename)
    return written


def source_for_stmt(source: str, stmt: ast.stmt) -> str:
    segment = ast.get_source_segment(source, stmt)
    if segment is None:
        segment = ast.unparse(stmt)
    return segment.strip()


def is_plot_style_expr(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Expr):
        return False
    call = stmt.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Attribute)
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "plt"
    )


def is_client_assignment(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Assign):
        return False
    target_names = {target.id for target in stmt.targets if isinstance(target, ast.Name)}
    return bool(target_names & {"qj", "START", "END"})


def split_preserved_leading_cell(nb: dict) -> None:
    if len(nb.get("cells", [])) < 2:
        return
    first_code_index = next((i for i, cell in enumerate(nb["cells"]) if cell.get("cell_type") == "code"), None)
    if first_code_index is None:
        return
    source = "".join(nb["cells"][first_code_index].get("source", []))
    if "QuantJourneyAPI" not in source or "def unwrap" not in source:
        return

    tree = ast.parse(source)
    buckets: dict[str, list[str]] = {
        "imports": [],
        "client": [],
        "response": [],
        "helpers": [],
        "workflow": [],
    }
    for stmt in tree.body:
        text = source_for_stmt(source, stmt)
        if isinstance(stmt, (ast.Import, ast.ImportFrom)) or is_plot_style_expr(stmt):
            buckets["imports"].append(text)
        elif is_client_assignment(stmt):
            buckets["client"].append(text)
        elif isinstance(stmt, ast.FunctionDef) and stmt.name in {"unwrap", "as_rows"}:
            buckets["response"].append(text)
        elif isinstance(stmt, ast.FunctionDef):
            buckets["helpers"].append(text)
        else:
            buckets["workflow"].append(text)

    replacement: list[dict] = []
    for title, key in [
        ("## Imports and Plot Style", "imports"),
        ("## QuantJourney Client", "client"),
        ("## Response Helpers", "response"),
        ("## Workflow Helpers", "helpers"),
        ("## Workflow", "workflow"),
    ]:
        if buckets[key]:
            replacement.extend([md(title), code("\n\n".join(buckets[key]))])
    nb["cells"] = nb["cells"][:first_code_index] + replacement + nb["cells"][first_code_index + 1 :]


def attach_preview_cells() -> None:
    for path in sorted(CANDIDATES.glob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        preserved_summary = None
        if path.name in PRESERVED_SOURCE_CANDIDATE_NAMES:
            preserved_summary = next(summary for name, _, summary in PRESERVED_SOURCE_CANDIDATES if name == path.name)
            title = path.stem.split("_", 1)[1].replace("_", " ").title() if "_" in path.stem else path.stem
            for cell in nb.get("cells", []):
                source = "".join(cell.get("source", []))
                if cell.get("cell_type") == "markdown":
                    for line in source.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("# "):
                            title = stripped[2:].strip()
                            break
                    break
            while title.startswith("QuantJourney SDK - "):
                title = title.removeprefix("QuantJourney SDK - ").strip()
        cleaned_cells = []
        for index, cell in enumerate(nb.get("cells", [])):
            source = "".join(cell.get("source", []))
            if preserved_summary and index == 0 and cell.get("cell_type") == "markdown":
                cleaned_cells.append(md(example_intro(title, preserved_summary)))
                continue
            if cell.get("cell_type") == "markdown" and source.lstrip().startswith(("## Run Output", "## Preview Chart")):
                continue
            if cell.get("cell_type") == "markdown" and "Prepared by QuantJourney" in source:
                continue
            if cell.get("cell_type") == "markdown" and "Generated run artifacts" in source:
                continue
            cleaned_cells.append(cell)
        nb["cells"] = cleaned_cells
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                source = strip_icons(source)
                cell["source"] = normalize_code_source(source).splitlines(keepends=True)
                cell["execution_count"] = None
                cell["outputs"] = []
            elif cell.get("cell_type") == "markdown":
                source = strip_icons("".join(cell.get("source", [])))
                cell["source"] = source.splitlines(keepends=True)
        normalize_core_setup_cells(nb, path)
        if path.name in PRESERVED_SOURCE_CANDIDATE_NAMES:
            split_preserved_leading_cell(nb)
        path.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")


def write_index(copied: list[tuple[str, str, str]], generated: list[str]) -> None:
    rows = []
    for name, category, summary in copied:
        rows.append((name, category, summary))
    for name, category, summary in PRESERVED_SOURCE_CANDIDATES:
        if (CANDIDATES / name).exists():
            rows.append((name, category, summary))
    for spec in ALL_GENERATED_SPECS:
        rows.append((spec.filename, spec.category, spec.summary))
    rows = sorted(rows, key=lambda row: row[0])
    lines = [
        "# QuantJourney SDK Example Catalog",
        "",
        "Flat example catalog for QuantJourney SDK workflows. Example notebooks are clean source files; generated chart artifacts are committed under `_output/` and indexed in `_output/manifest.json`. New examples use direct QuantJourney SDK connector calls plus transparent local analytics. Production systems can wrap the same workflows behind governed domain routes, tenant scopes and audit metadata.",
        "",
        "| Notebook | Output | Type | What it shows |",
        "|---|---|---|---|",
    ]
    for name, category, summary in rows:
        stem = Path(name).stem
        if stem == "01_authentication_methods":
            output_link = "none"
        elif stem == "02_market_data_basics":
            output_link = "[01](../_output/02_market_01.png), [02](../_output/02_market_02.png), [03](../_output/02_market_03.png)"
        else:
            output_link = f"[PNG](../_output/{stem}_output_01.png)"
        lines.append(f"| [{name}]({name}) | {output_link} | {category} | {summary} |")
    lines.extend(
        [
            "",
            "## Run",
            "",
            "```bash",
            "export QJ_API_KEY=\"qj_...\"",
            "jupyter lab _candidates",
            "```",
            "",
            "These examples use direct SDK calls. Missing tenant access or missing provider coverage should surface as normal API errors during execution.",
        ]
    )
    (CANDIDATES / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    clean_example_notebooks()
    copied = copy_existing()
    generated = write_generated(ALL_GENERATED_SPECS)
    attach_preview_cells()
    write_index(copied, generated)
    print(f"Copied {len(copied)} existing notebooks")
    print(f"Generated {len(generated)} example notebooks")
    print(f"Wrote {CANDIDATES / 'INDEX.md'}")


if __name__ == "__main__":
    main()
