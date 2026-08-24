# Customer Segmentation Streamlit Frontend — Enhanced

This version adds:

- Executive dashboard KPIs
- Key business insights
- Revenue concentration analysis
- RFM segment positioning
- Customer-level recommendations
- Segment-level recommendations
- Customer vs segment median comparison
- Segment RFM analysis
- Model validation dashboard
- K-selection charts when multiple K results are stored
- ARI stability metrics
- SQLite data explorer
- CSV downloads

## Folder structure

Keep these files together:

```text
customer_segmentation_streamlit/
├── app.py
├── customer_segmentation.db
└── requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Database requirements

The SQLite database should contain:

- `customers`
- `transactions`
- `model_results`

The `customers` table should contain at least:

- Customer ID
- Recency
- Frequency
- Monetary
- Cluster
- Segment

AOV is optional.

For the full K-selection page, store one row per K in `model_results` with:

- K
- Silhouette
- Calinski_Harabasz
- Davies_Bouldin

The ARI columns can be stored on the selected-model row.
