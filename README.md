# QuantJourney SDK Examples

This directory contains Python scripts and Jupyter notebooks demonstrating how to use the QuantJourney SDK.

## Prerequisites

- Python 3.10+
- Access to QuantJourney API (https://api.quantjourney.cloud)
- Demo credentials: `demo@quantjourney.cloud` / `demo123` / tenant: `default`

## Required Packages

```bash
pip install pandas numpy plotly
```

## Examples

### 1. Authentication Methods
**Files:** `01_authentication_methods.py`, `01_authentication_methods.ipynb`

Learn different ways to authenticate with the API:
- Direct token/API key
- Username/password login
- Environment variables
- Config file

### 2. Market Data Basics
**Files:** `02_market_data_basics.py`, `02_market_data_basics.ipynb`

Basic market data retrieval and visualization:
- Historical OHLCV prices
- Company information
- Multi-stock comparison (FAANG)
- Basic statistics (returns, volatility)
- Candlestick charts with volume
- Performance comparison charts

### 3. Economic/Macro Data
**Files:** `03_economic_data_macro.py`, `03_economic_data_macro.ipynb`

Macroeconomic analysis with FRED data:
- GDP and economic growth
- Inflation (CPI) with Fed target
- Unemployment rate
- Treasury yields (2Y, 10Y, 30Y)
- Federal Funds rate
- Yield curve analysis

### 4. Fundamental Analysis
**Files:** `04_fundamental_analysis.py`, `04_fundamental_analysis.ipynb`

Company fundamentals and financial statements:
- Company profiles
- Income statements and margins
- Balance sheet analysis
- Cash flow statements
- Key financial ratios (P/E, P/B, ROE, ROA)
- Peer comparison (FAANG)
- Valuation summary

### 5. Technical Analysis
**Files:** `05_technical_analysis.py`, `05_technical_analysis.ipynb`

Quantitative technical indicators:
- Moving averages (SMA 20/50/200, EMA 12/26)
- Bollinger Bands
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Volatility analysis (20-day, 60-day rolling)
- Risk metrics (Sharpe, Max Drawdown, VaR)

### 6. Portfolio Analysis
**Files:** `06_portfolio_analysis.py`, `06_portfolio_analysis.ipynb`

Portfolio construction and analytics:
- Multi-asset portfolio (9 stocks, diversified sectors)
- Performance tracking vs individual holdings
- Correlation matrix / heatmap
- Risk metrics (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Risk contribution by asset
- Monthly returns heatmap

### 7. Crypto & Alternative Data
**Files:** `07_crypto_alternative_data.py`, `07_crypto_alternative_data.ipynb`

Alternative data sources:
- **CCXT**: Cryptocurrency data from exchanges (Binance, etc.)
- **CBOE**: VIX volatility index
- **CFTC**: Commitment of Traders (COT) reports
- **Multpl**: Shiller P/E, S&P 500 metrics
- **FINRA**: Short interest data
- **SEC**: Company filings (10-K, 10-Q)
- **OpenFIGI**: Global instrument identifiers

## Usage

### Python Scripts
```bash
python 02_market_data_basics.py
```

### Jupyter Notebooks
```bash
jupyter notebook 02_market_data_basics.ipynb
```

### Quick Start
```python
import sys
sys.path.insert(0, '..')

from quantjourney.sdk import QuantJourneyAPI

# Connect
qj = QuantJourneyAPI(api_url="https://api.quantjourney.cloud")
qj.auth.login(
    email="demo@quantjourney.cloud",
    password="demo123",
    tenant_id="default"
)

# Fetch data
prices = qj.eod.get_historical_prices(
    symbol="AAPL",
    start_date="2024-01-01",
    end_date="2024-12-31",
    frequency="1d"
)
```

## Available Connectors

| Connector | Data Source | Examples |
|-----------|-------------|----------|
| `qj.eod` | EOD Historical Data | Prices, fundamentals |
| `qj.fmp` | Financial Modeling Prep | Profiles, financials, earnings |
| `qj.fred` | Federal Reserve (FRED) | GDP, CPI, unemployment, rates |
| `qj.ccxt` | Crypto exchanges | BTC, ETH, SOL from Binance, etc. |
| `qj.cboe` | CBOE | VIX, options data |
| `qj.cftc` | CFTC | COT reports, futures positioning |
| `qj.multpl` | Multpl | Shiller P/E, market metrics |
| `qj.finra` | FINRA | Short interest |
| `qj.sec` | SEC EDGAR | Company filings |
| `qj.openfigi` | OpenFIGI | Instrument identifiers |

## Notebook Features

Each Jupyter notebook includes:
- 📊 Interactive Plotly visualizations
- 📈 Professional charts (candlestick, heatmaps, subplots)
- 📋 Summary dashboards
- 💡 Practical quant analysis examples

## File Structure

```
_sdk_examples/
├── 01_authentication_methods.py    # Auth examples
├── 01_authentication_methods.ipynb
├── 02_market_data_basics.py        # Market data
├── 02_market_data_basics.ipynb
├── 03_economic_data_macro.py       # FRED/macro
├── 03_economic_data_macro.ipynb
├── 04_fundamental_analysis.py      # Fundamentals
├── 04_fundamental_analysis.ipynb
├── 05_technical_analysis.py        # Technical/quant
├── 05_technical_analysis.ipynb
├── 06_portfolio_analysis.py        # Portfolio
├── 06_portfolio_analysis.ipynb
├── 07_crypto_alternative_data.py   # Alt data
├── 07_crypto_alternative_data.ipynb
└── README.md
```

## Support

For issues or questions, contact: support@quantjourney.pro
