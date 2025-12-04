"""Utilities for fetching and storing cryptocurrency OHLCV data."""

from __future__ import annotations

import datetime as dt
import os
from typing import Iterable, List

import pandas as pd
import requests
from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, UniqueConstraint
from sqlalchemy.dialects.postgresql import insert

from .db import get_engine

# Placeholder endpoint; update when the exact FreeCryptoAPI path is confirmed.
BASE_URL = "https://api.freecryptoapi.com/v1/market/ohlcv"


def _ensure_table(metadata: MetaData) -> Table:
    """Create the crypto_prices table if it does not exist."""

    table = Table(
        "crypto_prices",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("timestamp", DateTime(timezone=True), nullable=False),
        Column("symbol", String, nullable=False),
        Column("open", Float),
        Column("high", Float),
        Column("low", Float),
        Column("close", Float),
        Column("volume", Float),
        Column("source", String, default="FreeCryptoAPI"),
        UniqueConstraint("timestamp", "symbol", name="uq_crypto_timestamp_symbol"),
    )
    return table


def fetch_crypto_history(symbol: str, days: int, api_key: str) -> pd.DataFrame:
    """Fetch daily OHLCV data for ``symbol`` for the past ``days`` days."""

    end = dt.datetime.utcnow().date()
    start = end - dt.timedelta(days=days)
    params = {
        "symbol": symbol,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "api_key": api_key,
    }
    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if "prices" not in data:
        raise ValueError("Unexpected API response: missing 'prices' field")

    df = pd.DataFrame(data["prices"])
    if df.empty:
        raise ValueError(f"No price data returned for symbol {symbol}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df["symbol"] = symbol.upper()
    return df[["symbol", "open", "high", "low", "close", "volume"]]


def save_crypto_to_db(df: pd.DataFrame) -> None:
    """Persist an OHLCV DataFrame into PostgreSQL with upsert semantics."""

    engine = get_engine()
    metadata = MetaData()
    crypto_prices = _ensure_table(metadata)
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


def load_crypto_from_db(symbols: Iterable[str], start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
    """Load OHLCV data for ``symbols`` between ``start_date`` and ``end_date``."""

    engine = get_engine()
    symbol_list: List[str] = [sym.upper() for sym in symbols]
    query = (
        "SELECT timestamp, symbol, open, high, low, close, volume "
        "FROM crypto_prices "
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
        return df

    df = df.pivot_table(
        index="timestamp",
        columns="symbol",
        values=["open", "high", "low", "close", "volume"],
    ).sort_index()
    return df
