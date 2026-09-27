"""Parity tests: the SQL RFM path must match the original pandas logic.

The dashboard's segments are a pure function of the R/F/M values, so if the
SQL-computed R/F/M equals the pandas-computed R/F/M for every customer, the
downstream K-Means segmentation is guaranteed to produce identical segment
counts. Both facts are asserted below.
"""

import sqlite3

import pandas as pd
import pytest

from rfm_pipeline import (
    DB_PATH,
    compute_rfm_pandas,
    compute_rfm_sql,
    run_segmentation,
)


@pytest.fixture(scope="module")
def conn():
    if not DB_PATH.exists():
        pytest.skip(f"database not found: {DB_PATH}")
    connection = sqlite3.connect(DB_PATH)
    yield connection
    connection.close()


def test_sql_rfm_matches_pandas(conn):
    """SQL R/F/M equals pandas R/F/M per customer."""
    sql = compute_rfm_sql(conn)[["Customer ID", "Recency", "Frequency", "Monetary"]]
    pandas_rfm = compute_rfm_pandas(conn)

    assert len(sql) == len(pandas_rfm)
    assert set(sql["Customer ID"]) == set(pandas_rfm["Customer ID"])

    merged = pandas_rfm.merge(sql, on="Customer ID", suffixes=("_pd", "_sql"))

    pd.testing.assert_series_equal(
        merged["Recency_pd"], merged["Recency_sql"], check_names=False
    )
    pd.testing.assert_series_equal(
        merged["Frequency_pd"], merged["Frequency_sql"], check_names=False
    )
    # Monetary is a float sum; allow only floating-point rounding noise.
    assert (merged["Monetary_pd"] - merged["Monetary_sql"]).abs().max() < 1e-6


def test_segment_counts_match(conn):
    """The SQL path yields the same segment counts as the pandas path."""
    seg_sql = run_segmentation(compute_rfm_sql(conn))
    seg_pandas = run_segmentation(compute_rfm_pandas(conn))

    counts_sql = seg_sql["Segment"].value_counts().sort_index()
    counts_pandas = seg_pandas["Segment"].value_counts().sort_index()

    pd.testing.assert_series_equal(counts_sql, counts_pandas)
