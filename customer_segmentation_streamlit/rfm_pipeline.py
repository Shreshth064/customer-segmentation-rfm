"""RFM aggregation and customer segmentation pipeline.

This module computes the Recency / Frequency / Monetary (RFM) features in
**SQL using window functions** instead of the original in-memory pandas
``groupby`` pass, and then runs the unchanged K-Means segmentation on top.

Engine choice: SQLite. The cleaned transaction table already lives in the
project's SQLite database (``customer_segmentation.db``) and the Streamlit app
already talks to it via ``sqlite3``, so the SQL RFM path adds zero new
dependencies. SQLite >= 3.25 (bundled with modern Python) supports the window
functions used here, including ``NTILE``.

The pandas reference implementation (``compute_rfm_pandas``) is kept so tests
can assert that the SQL path produces byte-for-byte identical R/F/M values, and
therefore identical downstream segment counts.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DB_PATH = Path(__file__).parent / "customer_segmentation.db"


# ============================================================
# SQL RFM (window functions)
# ============================================================

# Recency / Frequency / Monetary computed entirely in SQL.
#
# Pandas equivalent this replaces (notebooks/ml.ipynb, cell 9):
#
#     reference_date = df_sales["InvoiceDate"].max() + pd.Timedelta(days=1)
#     rfm = df_sales.groupby("Customer ID").agg(
#         Recency  =("InvoiceDate", lambda x: (reference_date - x.max()).days),
#         Frequency=("Invoice", "nunique"),
#         Monetary =("Revenue", "sum"),
#     ).reset_index()
#
RFM_SQL = """
WITH invoices AS (
    -- Roll transaction LINES up to the INVOICE grain.
    -- Frequency = number of DISTINCT invoices, but SQL does not allow
    -- COUNT(DISTINCT ...) as a window function. Deduping to one row per
    -- (customer, invoice) here lets us use a plain COUNT(*) window aggregate
    -- below, so all three R/F/M aggregates become genuine window functions.
    SELECT
        "Customer ID"    AS customer_id,
        Invoice,
        MAX(InvoiceDate) AS invoice_date,     -- one timestamp per invoice
        SUM(Revenue)     AS invoice_revenue   -- sum of the invoice's line items
    FROM transactions
    GROUP BY "Customer ID", Invoice
),
customer_rfm AS (
    -- Collapse to one row per customer using window aggregates PARTITIONed
    -- by customer. Every window value is identical within a customer, so
    -- SELECT DISTINCT yields exactly one row per customer.
    SELECT DISTINCT
        customer_id,
        -- Recency: whole days from the customer's most recent invoice to the
        -- global reference date (latest invoice across ALL customers + 1 day).
        -- MAX(...) OVER () with an empty window spans the whole table = the
        -- global reference point. CAST(... AS INTEGER) truncates the
        -- fractional day, matching pandas Timedelta(...).days.
        CAST(
            (MAX(julianday(invoice_date)) OVER () + 1.0)
            - MAX(julianday(invoice_date)) OVER (PARTITION BY customer_id)
            AS INTEGER
        )                                                    AS Recency,
        COUNT(*)             OVER (PARTITION BY customer_id) AS Frequency,
        SUM(invoice_revenue) OVER (PARTITION BY customer_id) AS Monetary
    FROM invoices
)
SELECT
    customer_rfm.customer_id AS "Customer ID",
    Recency,
    Frequency,
    Monetary,
    -- RFM SCORES via NTILE: 5 equal-sized buckets across the population.
    -- Recency is reverse-scored so the most-recent customers (smallest
    -- Recency) land in the best bucket (5). Frequency/Monetary score
    -- ascending so bigger spenders/buyers land in the best bucket (5).
    -- These scores are an ADDITIONAL RFM feature; the dashboard segments
    -- still come from the K-Means step below, not from these buckets.
    NTILE(5) OVER (ORDER BY Recency DESC)  AS R_Score,
    NTILE(5) OVER (ORDER BY Frequency ASC) AS F_Score,
    NTILE(5) OVER (ORDER BY Monetary ASC)  AS M_Score
FROM customer_rfm
ORDER BY customer_id
"""


def compute_rfm_sql(conn: sqlite3.Connection) -> pd.DataFrame:
    """Compute R/F/M (and NTILE R/F/M scores) from ``transactions`` via SQL.

    Returns a DataFrame with columns:
    ``Customer ID, Recency, Frequency, Monetary, R_Score, F_Score, M_Score``.
    """
    return pd.read_sql(RFM_SQL, conn)


# ============================================================
# Pandas RFM (reference implementation, used by tests)
# ============================================================

def compute_rfm_pandas(conn: sqlite3.Connection) -> pd.DataFrame:
    """Original in-memory RFM aggregation, kept as the correctness oracle.

    Reads the same ``transactions`` table the SQL path reads so both paths
    start from an identical source.
    """
    df = pd.read_sql('SELECT * FROM transactions', conn)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("Customer ID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
        .sort_values("Customer ID")
        .reset_index(drop=True)
    )

    return rfm


# ============================================================
# Segmentation (unchanged K-Means logic, lifted from the notebook)
# ============================================================

# Cluster-id -> business segment name, exactly as in notebooks/ml.ipynb.
CLUSTER_NAMES = {
    0: "Inactive / Low-Value",
    1: "At-Risk / Mid-Value",
    2: "Recent / Emerging",
    3: "Champions / High-Value",
}

RFM_FEATURES = ["Recency_log", "Frequency_log", "Monetary_log"]


def run_segmentation(rfm: pd.DataFrame, optimal_k: int = 4) -> pd.DataFrame:
    """Log-transform, standardize, cluster and label customers.

    This reproduces the notebook's clustering step verbatim so that, given
    identical R/F/M inputs, it yields identical clusters and segment counts.
    """
    rfm = rfm.copy()

    rfm["Recency_log"] = np.log1p(rfm["Recency"])
    rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[RFM_FEATURES])

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    rfm["Cluster"] = kmeans.fit_predict(X)

    rfm["Segment"] = rfm["Cluster"].map(CLUSTER_NAMES)

    rfm["AOV"] = np.where(
        rfm["Frequency"] > 0,
        rfm["Monetary"] / rfm["Frequency"],
        np.nan,
    )

    return rfm


def build_customers_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """Full SQL-RFM -> segmentation pipeline, matching the notebook's output."""
    rfm = compute_rfm_sql(conn)
    return run_segmentation(rfm)


def main() -> None:
    """Rebuild the ``customers`` table from the SQL RFM path."""
    if not DB_PATH.exists():
        raise SystemExit(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        customers = build_customers_table(conn)
        customers.to_sql("customers", conn, if_exists="replace", index=False)
        conn.commit()
        print(f"Wrote {len(customers):,} rows to customers via SQL RFM path.")
        print(customers["Segment"].value_counts().to_string())
    finally:
        conn.close()


if __name__ == "__main__":
    main()
