# Cryptocurrency Price Dynamics Pipeline

This project demonstrates a fully reproducible Python workflow for cleaning and
merging cryptocurrency, macroeconomic, and sentiment signals to forecast
next-day price movements. The included script uses synthetic data so it can run
offline without paid services. You can swap in your own CSV exports or free
APIs to power the same pipeline.

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
2. Run the pipeline (uses 180 synthetic days by default):
   ```bash
   python src/pipeline.py
   ```
3. Adjust parameters as needed:
   ```bash
   python src/pipeline.py --days 365 --test-size 0.25 --n-jobs 4
   ```

The script prints the tail of the merged dataset and evaluation metrics for each
model on a hold-out test split.

## Adapting to Real Data
- Replace the synthetic generation functions in `src/pipeline.py` with loaders
  that read your CSV exports or free API responses.
- Keep the `merge_datasets` and `engineer_features` steps to preserve the
  cleaning logic and feature set.
- Add or remove models inside `build_models` to experiment with additional
  algorithms.
