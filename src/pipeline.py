"""Predictive modeling pipeline for cryptocurrency price dynamics.

This script builds an end-to-end workflow for assembling synthetic cryptocurrency,
macroeconomic, and sentiment datasets; cleaning and aligning timestamps;
engineering features; and training several baseline machine learning models to
forecast next-day price movements.

The code is designed to be easily extended to real data sources. Replace the
synthetic generation functions with calls to your preferred free APIs or local
CSV files, and keep the cleaning/merging/modeling steps intact.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_ingestion.crypto_loader import fetch_crypto_history, load_crypto_from_db, save_crypto_to_db

load_dotenv()

# Set a global random seed for reproducibility across all synthetic data and models.
RNG = np.random.default_rng(seed=42)


@dataclass
class Dataset:
    """Bundle of features and target labels for model training."""

    features: pd.DataFrame
    target: pd.Series


@dataclass
class ModelResult:
    """Simple container for trained models and their evaluation metrics."""

    model_name: str
    model: object
    metrics: Dict[str, float]


def generate_synthetic_crypto_data(days: int = 180) -> pd.DataFrame:
    """Create a synthetic OHLCV dataset to mimic crypto price movements.

    Args:
        days: Number of calendar days to simulate.

    Returns:
        DataFrame with datetime index and OHLCV columns.
    """

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    # Simulate a noisy geometric random walk for closing prices.
    price = 20000 * np.exp(np.cumsum(RNG.normal(0, 0.01, size=days)))
    close = price
    open_ = close * (1 + RNG.normal(0, 0.002, size=days))
    high = np.maximum(open_, close) * (1 + RNG.normal(0.001, 0.003, size=days))
    low = np.minimum(open_, close) * (1 - RNG.normal(0.001, 0.003, size=days))
    volume = RNG.lognormal(mean=12, sigma=0.4, size=days)

    crypto_df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    crypto_df.index.name = "timestamp"
    return crypto_df


def generate_synthetic_macro_data(days: int = 180) -> pd.DataFrame:
    """Simulate macroeconomic indicators aligned to the same calendar days.

    The indicators include a weekly risk-free rate proxy and a monthly inflation
    signal. Forward filling expands low-frequency points to daily resolution.
    """

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    # Weekly rate changes and monthly inflation estimates.
    weekly_idx = dates[::7]
    monthly_idx = dates[::30]
    weekly_rate = pd.Series(RNG.normal(0.02, 0.002, len(weekly_idx)), index=weekly_idx)
    monthly_cpi = pd.Series(RNG.normal(0.005, 0.0015, len(monthly_idx)), index=monthly_idx)

    macro_df = pd.DataFrame(index=dates)
    macro_df["risk_free_rate"] = weekly_rate.reindex(dates).ffill()
    macro_df["inflation_rate"] = monthly_cpi.reindex(dates).ffill()
    macro_df.index.name = "timestamp"
    return macro_df


def generate_synthetic_sentiment(days: int = 180) -> pd.DataFrame:
    """Produce daily sentiment signals representing online discussions.

    Sentiment is modeled as a smoothed random walk constrained to [-1, 1], with
    random spikes to emulate viral events.
    """

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    base_signal = np.cumsum(RNG.normal(0, 0.02, size=days))
    spikes = RNG.choice([0, 0, 0.5, -0.5], size=days, p=[0.85, 0.1, 0.03, 0.02])
    sentiment_score = np.tanh(base_signal + spikes)

    sentiment_df = pd.DataFrame({"sentiment": sentiment_score}, index=dates)
    sentiment_df.index.name = "timestamp"
    return sentiment_df


def merge_datasets(crypto: pd.DataFrame, macro: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """Align and merge datasets on the timestamp index.

    Missing values are forward filled to preserve continuity across different
    sampling frequencies.
    """

    merged = crypto.join([macro, sentiment], how="left").sort_index()
    merged.ffill(inplace=True)
    return merged


def engineer_features(merged: pd.DataFrame) -> Dataset:
    """Create predictive features and the next-day return target.

    Features include:
    * Daily returns and next-day returns (target)
    * Rolling moving averages and volatility
    * Volume momentum
    * Rolling sentiment
    * Macro trends via day-over-day differences
    """

    df = merged.copy()

    # Price-based features.
    df["return"] = df["close"].pct_change()
    df["high_low_spread"] = (df["high"] - df["low"]) / df["close"]
    df["volatility_7d"] = df["return"].rolling(window=7).std()
    df["ma_7"] = df["close"].rolling(window=7).mean()
    df["ma_30"] = df["close"].rolling(window=30).mean()
    df["ma_ratio"] = df["ma_7"] / df["ma_30"]

    # Volume and sentiment momentum.
    volume_std = df["volume"].rolling(14).std()
    df["volume_z"] = (df["volume"] - df["volume"].rolling(14).mean()) / volume_std.replace(0, 1)
    df["sentiment_7d"] = df["sentiment"].rolling(window=7).mean()

    # Macroeconomic differences to capture trend direction.
    df["risk_free_rate_change"] = df["risk_free_rate"].diff()
    df["inflation_change"] = df["inflation_rate"].diff()

    # Target: predict the next day's return.
    df["target_return"] = df["return"].shift(-1)

    # Drop rows with NaN after feature construction.
    df = df.dropna()

    feature_cols = [
        "high_low_spread",
        "volatility_7d",
        "ma_ratio",
        "volume_z",
        "sentiment_7d",
        "risk_free_rate_change",
        "inflation_change",
    ]
    features = df[feature_cols]
    target = df["target_return"]
    return Dataset(features=features, target=target)


def build_models(n_jobs: int = -1) -> Dict[str, Pipeline]:
    """Configure baseline regression models with preprocessing pipelines."""

    numeric_features = [
        "high_low_spread",
        "volatility_7d",
        "ma_ratio",
        "volume_z",
        "sentiment_7d",
        "risk_free_rate_change",
        "inflation_change",
    ]

    preprocessor = ColumnTransformer(
        [
            ("scale", StandardScaler(), numeric_features),
        ],
        remainder="passthrough",
    )

    models: Dict[str, Pipeline] = {
        "linear_regression": Pipeline([("prep", preprocessor), ("model", LinearRegression())]),
        "random_forest": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Compute standard regression metrics for a trained model."""

    preds = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


def _flatten_crypto_columns(crypto_df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """Flatten a MultiIndex column DataFrame to a single symbol OHLCV frame."""

    if crypto_df.empty:
        return crypto_df

    if isinstance(crypto_df.columns, pd.MultiIndex):
        primary_symbol = symbols[0].upper()
        if primary_symbol not in crypto_df.columns.get_level_values(1):
            raise RuntimeError(
                f"Primary symbol '{primary_symbol}' not found in returned data."
            )
        flattened = crypto_df.xs(primary_symbol, level=1, axis=1)
    else:
        flattened = crypto_df

    flattened = flattened.sort_index()
    if getattr(flattened.index, "tz", None) is not None:
        flattened.index = flattened.index.tz_localize(None)
    flattened.index.name = "timestamp"
    flattened.columns = [col.lower() for col in flattened.columns]
    expected_cols = {"open", "high", "low", "close", "volume"}
    missing = expected_cols.difference(flattened.columns)
    if missing:
        raise RuntimeError(
            "Missing expected OHLCV columns after flattening: " + ", ".join(sorted(missing))
        )
    ordered_cols = ["open", "high", "low", "close", "volume"]
    return flattened[ordered_cols]


def run_pipeline(
    days: int = 180,
    test_size: float = 0.2,
    n_jobs: int = -1,
    data_source: str = "api",
    symbols: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full pipeline (API or synthetic) and return merged data and metrics."""

    symbols = symbols or ["BTC", "ETH"]

    if data_source not in {"api", "synthetic"}:
        raise ValueError("data_source must be either 'api' or 'synthetic'")

    if data_source == "api":
        api_key = os.getenv("COINGECKO_API_KEY")
        if not api_key:
            raise RuntimeError(
                "COINGECKO_API_KEY environment variable not set. "
                "Add it to your .env or environment, or run with --data-source synthetic."
            )

        # Fetch and persist latest prices before loading for modeling.
        for sym in symbols:
            df = fetch_crypto_history(sym, days=days, api_key=api_key)
            save_crypto_to_db(df)

        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.Timedelta(days=days)
        crypto_raw = load_crypto_from_db(symbols, start, end)
        crypto_df = _flatten_crypto_columns(crypto_raw, symbols)
    else:
        crypto_df = generate_synthetic_crypto_data(days)

    macro_df = generate_synthetic_macro_data(days)
    sentiment_df = generate_synthetic_sentiment(days)

    merged = merge_datasets(crypto_df, macro_df, sentiment_df)
    dataset = engineer_features(merged)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features, dataset.target, test_size=test_size, shuffle=False
    )

    models = build_models(n_jobs=n_jobs)
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results.append(ModelResult(model_name=name, model=model, metrics=metrics))

    # Convert results to a DataFrame for easy inspection and return both.
    metrics_df = pd.DataFrame(
        [
            {
                "model": res.model_name,
                **res.metrics,
            }
            for res in results
        ]
    ).set_index("model")

    return merged, metrics_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predictive modeling pipeline for cryptocurrency price dynamics"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of calendar days of data to fetch or simulate (default: 180)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples to allocate to test set (default: 0.2)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallelism for tree models; -1 uses all available cores",
    )
    parser.add_argument(
        "--data-source",
        choices=["api", "synthetic"],
        default="api",
        help="Use 'api' to fetch live OHLCV data (default) or 'synthetic' to simulate offline.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC", "ETH"],
        help="Crypto tickers to ingest when using the API; the first symbol is used for modeling.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    merged, metrics = run_pipeline(
        days=args.days,
        test_size=args.test_size,
        n_jobs=args.n_jobs,
        data_source=args.data_source,
        symbols=args.symbols,
    )

    print("Merged dataset sample (last 5 rows):")
    print(merged.tail())
    print("\nModel performance on hold-out set:")
    print(metrics)


if __name__ == "__main__":
    main()
