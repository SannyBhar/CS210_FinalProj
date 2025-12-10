"""Streamlit UI for running the cryptocurrency price dynamics pipeline."""

from __future__ import annotations

import os
from typing import List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pipeline import run_pipeline

# Load environment variables from a local .env if present.
load_dotenv()


def _parse_symbols(symbol_text: str) -> List[str]:
    """Normalize a comma-separated symbol string into a list."""

    symbols = [sym.strip().upper() for sym in symbol_text.split(",") if sym.strip()]
    if not symbols:
        raise ValueError("Provide at least one symbol (e.g., BTC or BTC,ETH).")
    return symbols


def _set_env_var(name: str, value: str | None) -> None:
    """Set an environment variable only when a value is provided."""

    if value:
        os.environ[name] = value


def _display_results(merged: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Render charts and tables from the pipeline output."""

    st.success("Pipeline completed successfully.")
    st.subheader("Model performance")
    st.dataframe(metrics.reset_index().rename(columns={"index": "model"}), use_container_width=True)

    st.subheader("Price trend (primary symbol)")
    if "close" in merged.columns:
        st.line_chart(merged[["close"]])
    else:
        st.info("Close price column not found; skipping price chart.")

    st.subheader("Merged dataset preview")
    st.dataframe(merged.tail(25), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Crypto Pipeline", layout="wide")
    st.title("Cryptocurrency Price Dynamics Pipeline")
    st.write(
        "Configure a run below to fetch/ingest OHLCV data, merge synthetic macro and "
        "sentiment signals, engineer features, and evaluate baseline models."
    )

    key_set = bool(os.getenv("COINGECKO_API_KEY"))
    db_set = bool(os.getenv("DATABASE_URL"))
    st.caption(
        f"COINGECKO_API_KEY: {'✅ set' if key_set else '❌ missing'} · "
        f"DATABASE_URL: {'✅ set' if db_set else '❌ missing'}"
    )

    with st.sidebar:
        st.header("Run configuration")
        data_source = st.radio(
            "Data source",
            options=["api", "synthetic"],
            index=1,
            help="Use synthetic data for quick offline runs or API to fetch and store OHLCV.",
        )
        days = st.slider("Days of history", min_value=60, max_value=365, value=180, step=30)
        test_size = st.slider("Test size fraction", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        n_jobs = st.number_input(
            "Parallel jobs (tree models)",
            min_value=-1,
            max_value=32,
            value=-1,
            help="-1 uses all cores; increase for faster RandomForest training.",
        )
        symbols_text = st.text_input("Symbols (comma-separated)", value="BTC,ETH")

        if data_source == "api":
            st.divider()
            st.caption("API mode uses credentials from your environment (.env).")
            api_key_present = bool(os.getenv("COINGECKO_API_KEY"))
            db_url_present = bool(os.getenv("DATABASE_URL"))
            st.write(f"COINGECKO_API_KEY: {'✅ set' if api_key_present else '⚠️ missing'}")
            st.write(f"DATABASE_URL: {'✅ set' if db_url_present else '⚠️ missing'}")

        run_clicked = st.button("Run pipeline", type="primary")

    if not run_clicked:
        st.info("Adjust the settings in the sidebar and click **Run pipeline** to start.")
        return

    try:
        symbols = _parse_symbols(symbols_text)
    except ValueError as exc:
        st.error(str(exc))
        return

    if data_source == "api":
        if not os.getenv("COINGECKO_API_KEY") or not os.getenv("DATABASE_URL"):
            st.error("In API mode, set COINGECKO_API_KEY and DATABASE_URL in your environment or .env file.")
            return

    with st.spinner("Running pipeline... this may take a moment."):
        try:
            merged, metrics = run_pipeline(
                days=days,
                test_size=test_size,
                n_jobs=n_jobs,
                data_source=data_source,
                symbols=symbols,
            )
        except Exception as exc:  # Broad catch to surface errors in the UI.
            st.error(f"Pipeline failed: {exc}")
            return

    _display_results(merged, metrics)


if __name__ == "__main__":
    main()
