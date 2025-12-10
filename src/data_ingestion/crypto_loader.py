"""Utilities for fetching and storing cryptocurrency OHLCV data via CoinGecko."""

from __future__ import annotations

import os
import datetime as dt
from typing import Iterable, List

import pandas as pd
import requests
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import insert

from .db import get_engine

# Configurable API base and timeout (does NOT read the API key)
API_BASE = os.getenv("COINGECKO_API_URL", "https://api.coingecko.com/api/v3")
REQUEST_TIMEOUT = int(os.getenv("COINGECKO_API_TIMEOUT", "15"))

# Symbol → CoinGecko ID mapping
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "DOGE": "dogecoin",
}


def _get_coin_id(symbol: str) -> str:
    coin_id = COINGECKO_IDS.get(symbol.upper())
    if not coin_id:
        raise ValueError(
            f"Unsupported symbol '{symbol}'. Add it to COINGECKO_IDS mapping."
        )
    return coin_id


def _ensure_table(metadata: MetaData, table_name: str) -> Table:
    """Create the crypto_prices table if it does not exist."""
    table = Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("timestamp", DateTime(timezone=True), nullable=False),
        Column("symbol", String, nullable=False),
        Column("open", Float),
        Column("high", Float),
        Column("low", Float),
        Column("close", Float),
        Column("volume", Float),
        Column("source", String, default="CoinGecko"),
        UniqueConstraint("timestamp", "symbol", name=f"uq_{table_name}_timestamp_symbol"),
    )
    return table


def _handle_response(response: requests.Response):
    """Validate a CoinGecko response and return parsed JSON."""
    if not response.ok:
        raise RuntimeError(
            f"CoinGecko API request failed with status {response.status_code}: {response.text}"
        )

    data = response.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"CoinGecko API error: {data.get('error')}")
    return data


def _fetch_ohlc(coin_id: str, days: int, api_key: str) -> pd.DataFrame:
    """Fetch OHLC data from /coins/{id}/ohlc."""
    url = f"{API_BASE}/coins/{coin_id}/ohlc"
    params = {
        "vs_currency": "usd",
        "days": days,
    }
    headers = {
        "x-cg-demo-api-key": api_key
    }

    resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    data = _handle_response(resp)

    if not isinstance(data, list) or not data:
        raise RuntimeError(
            f"Unexpected OHLC response structure for {coin_id}; expected non-empty list, got {type(data)}"
        )

    df = pd.DataFrame(data, columns=["timestamp_ms", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.drop(columns=["timestamp_ms"]).set_index("timestamp").sort_index()
    return df


def _fetch_volumes(coin_id: str, days: int, api_key: str) -> pd.DataFrame:
    """Fetch volume data from /coins/{id}/market_chart."""
    url = f"{API_BASE}/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
    }
    headers = {
        "x-cg-demo-api-key": api_key
    }

    resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    data = _handle_response(resp)

    if not isinstance(data, dict) or "total_volumes" not in data:
        raise RuntimeError(
            f"Unexpected market_chart response structure for {coin_id}; "
            f"found: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    volumes = data.get("total_volumes") or []
    if not volumes:
        raise RuntimeError(f"No volume data returned for {coin_id}")

    df = pd.DataFrame(volumes, columns=["timestamp_ms", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.drop(columns=["timestamp_ms"]).set_index("timestamp").sort_index()
    return df


def fetch_crypto_history(symbol: str, days: int, api_key: str) -> pd.DataFrame:
    """
    Public entrypoint: fetch daily OHLCV data for ``symbol`` via CoinGecko.

    ``api_key`` is passed in from the caller (e.g., pipeline.run_pipeline), so this
    module does NOT depend on environment variables at import time.
    """
    if days <= 0:
        raise ValueError("days must be positive.")

    coin_id = _get_coin_id(symbol)

    ohlc_df = _fetch_ohlc(coin_id, days, api_key)
    volume_df = _fetch_volumes(coin_id, days, api_key)

    merged = ohlc_df.join(volume_df, how="left")
    merged["volume"] = merged["volume"].fillna(0.0)
    merged["symbol"] = symbol.upper()
    merged.index.name = "timestamp"
    return merged[["symbol", "open", "high", "low", "close", "volume"]]


def save_crypto_to_db(df: pd.DataFrame, table_name: str = "crypto_prices") -> None:
    """Persist an OHLCV DataFrame into PostgreSQL with upsert semantics."""
    if df.empty:
        raise ValueError("Cannot save empty dataframe to the database.")

    engine = get_engine()
    metadata = MetaData()
    crypto_prices = _ensure_table(metadata, table_name)
    metadata.create_all(engine)

    records = df.reset_index().to_dict(orient="records")
    stmt = insert(crypto_prices).values(records)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=[crypto_prices.c.timestamp, crypto_prices.c.symbol],
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
            "source": stmt.excluded.source,
        },
    )
    with engine.begin() as conn:
        conn.execute(upsert_stmt)


def load_crypto_from_db(
    symbols: Iterable[str],
    start_date: dt.datetime,
    end_date: dt.datetime,
    table_name: str = "crypto_prices",
) -> pd.DataFrame:
    """Load OHLCV data for ``symbols`` between ``start_date`` and ``end_date``."""
    engine = get_engine()
    symbol_list: List[str] = [sym.upper() for sym in symbols]

    query = (
        "SELECT timestamp, symbol, open, high, low, close, volume "
        f"FROM {table_name} "
        "WHERE timestamp >= %(start)s AND timestamp <= %(end)s "
        "AND symbol = ANY(%(symbols)s)"
    )
    params = {
        "start": start_date,
        "end": end_date,
        "symbols": symbol_list,
    }

    df = pd.read_sql(query, engine, params=params, parse_dates=["timestamp"])
    if df.empty:
        raise RuntimeError("No crypto price data found in the requested range/symbols.")

    # Ensure naive UTC timestamps for downstream compatibility.
    if getattr(df["timestamp"].dt, "tz", None) is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    df = df.pivot_table(
        index="timestamp",
        columns="symbol",
        values=["open", "high", "low", "close", "volume"],
    ).sort_index()
    return df