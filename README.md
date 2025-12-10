# Cryptocurrency Price Dynamics Pipeline

This project demonstrates a fully reproducible Python workflow for cleaning and
merging cryptocurrency, macroeconomic, and sentiment signals to forecast
next-day price movements. The pipeline now supports live OHLCV ingestion from
CoinGecko (using environment variables for secrets) and still offers a
synthetic mode for offline exploration.

## Features
- Synthetic generators for OHLCV prices, macro indicators, and sentiment scores
- Timestamp alignment, forward-filling, and feature engineering (returns,
  moving averages, volatility, macro deltas, and sentiment trends)
- Baseline models: Linear Regression, Random Forest, and Gradient Boosting
- Reproducible results via fixed random seeds

## Quickstart
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Start PostgreSQL locally and create the database referenced by `DATABASE_URL`
   (example uses `crypto_project`).
3. Copy the environment template and add your credentials (never commit your
   filled-in `.env` file):
   ```bash
   cp .env.example .env
   # edit .env to set DATABASE_URL and COINGECKO_API_KEY
   ```
4. Run the pipeline with live API ingestion (default data source is `api`):
   ```bash
   python -m src.pipeline --days 180 --symbols BTC ETH
   ```
   The first symbol in `--symbols` is used for modeling after prices are stored
   in PostgreSQL.
5. Prefer an offline run? Switch to synthetic data:
   ```bash
   python -m src.pipeline --data-source synthetic --days 180
   ```
6. Adjust other parameters as needed:
   ```bash
   python -m src.pipeline --days 365 --test-size 0.25 --n-jobs 4 --data-source api
   ```

The script prints the tail of the merged dataset and evaluation metrics for each
model on a hold-out test split.

## Streamlit Frontend
Run an interactive UI to configure parameters and inspect results:
```bash
streamlit run src/streamlit_app.py
```
If you choose the `api` data source, make sure `DATABASE_URL` and `COINGECKO_API_KEY`
are set in your environment (they are read automatically; no UI input is needed).
Optional: override the REST endpoint with `COINGECKO_API_URL` and request timeout with
`COINGECKO_API_TIMEOUT` if you are testing against a mock server.

## Notes on Secrets and Data Sources
- `.env.example` documents the required environment variables. Keep your API key
  and database credentials in a private `.env` file that is excluded from git.
- `src/data_ingestion/crypto_loader.py` contains the ingestion helpers for
  CoinGecko. Extend `COINGECKO_IDS` to support additional symbols.
- Macro and sentiment data remain synthetic placeholders; swap in your preferred
  loaders while keeping `merge_datasets` and `engineer_features` intact.
