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

To rebuild the example catalog and regenerate chart artifacts for `_candidates/*.ipynb` notebooks:

```bash
export QJ_API_KEY="qj_..."
PYTHONPATH=/path/to/quantjourney-sdk python scripts/build_candidate_notebooks.py
PYTHONPATH=/path/to/quantjourney-sdk python scripts/run_candidate_files.py
```

## Example Outputs

The canonical public catalog is flat:

- source notebooks: [`_candidates`](./_candidates/INDEX.md)
- generated charts: [`_output`](./_output)
- output manifest: [`_output/manifest.json`](./_output/manifest.json)

Most examples have one matching `_output/<notebook>_output_01.png` file. Some examples intentionally have no chart output, and some have multiple dedicated chart files; see `_output/manifest.json`.

## Example Catalog

The [`_candidates`](./_candidates/INDEX.md) folder contains QuantJourney SDK examples: core notebooks `01-13` plus institutional workflows for construction, liquidity, capacity, stress, attribution, regulatory signals, options overlays and end-to-end PM reporting. The notebooks are clean source files with direct SDK calls; generated chart artifacts live in [`_output`](./_output) and are indexed by [`_output/manifest.json`](./_output/manifest.json). Examples use SDK calls such as `qj.eod`, `qj.fmp`, `qj.sec`, `qj.cboe`, `qj.cftc`, `qj.ff`, `qj.bt` and `qj.analytics`, then keep portfolio/risk calculations visible in pandas/numpy. Production systems can wrap the same workflows behind governed domain routes, tenant scopes and audit metadata.

Notebooks `11-19` focus on core institutional data primitives: SEC filings, FINRA short interest, OpenFIGI identity, adjustment semantics, domain route discovery, global macro sources, volatility feeds, index universe construction and lineage/audit packets.

Notebooks `80-88` focus on multi-source institutional workflows: evidence packets, pre-trade capacity, PEAD research, PM risk briefs, crowding flows, cross-asset macro scenarios, macro regime allocation, inflation shocks and COT positioning.

To rebuild the catalog:

```bash
export QJ_API_KEY="qj_..."
PYTHONPATH=/path/to/quantjourney-sdk python scripts/build_candidate_notebooks.py
PYTHONPATH=/path/to/quantjourney-sdk python scripts/run_candidate_files.py
```

`scripts/run_candidate_files.py` processes every `_candidates/*.ipynb` file and writes chart artifacts to `_output/`, plus `_output/manifest.json`.

## Source Notebooks

The canonical examples for public browsing live in `_candidates`. The `notebooks/` tree is kept as source material for core and buy-side notebooks, but generated chart artifacts are centralized in `_output`.

## Output Manifest

`_output/manifest.json` records the example notebooks and generated PNG files.

## Data and Licensing

QuantJourney is a software and API control-plane layer. Market data access depends on your own QuantJourney tenant configuration and any required provider licenses or BYOL credentials.
