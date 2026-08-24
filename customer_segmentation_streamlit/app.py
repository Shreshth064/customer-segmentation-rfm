from pathlib import Path
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# DATABASE
# ============================================================

DB_PATH = Path(__file__).parent / "customer_segmentation.db"


@st.cache_resource
def get_connection():
    if not DB_PATH.exists():
        return None

    return sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
    )


@st.cache_data
def load_table(table_name):
    conn = get_connection()

    if conn is None:
        return pd.DataFrame()

    allowed_tables = {
        "customers",
        "transactions",
        "model_results",
    }

    if table_name not in allowed_tables:
        raise ValueError("Invalid table name.")

    return pd.read_sql(
        f'SELECT * FROM "{table_name}"',
        conn,
    )


conn = get_connection()

if conn is None:
    st.error(
        "customer_segmentation.db was not found. "
        "Place the database in the same folder as app.py."
    )
    st.stop()


customers = load_table("customers")
transactions = load_table("transactions")
model_results = load_table("model_results")


if customers.empty:
    st.error(
        "The customers table is empty. Run the notebook first "
        "to populate the database."
    )
    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

# Make sure numeric fields are numeric.
for col in ["Recency", "Frequency", "Monetary", "AOV", "Cluster"]:
    if col in customers.columns:
        customers[col] = pd.to_numeric(
            customers[col],
            errors="coerce",
        )

customers["Segment"] = customers["Segment"].fillna("Unknown")

total_customers = len(customers)
total_revenue = customers["Monetary"].sum()
avg_customer_value = customers["Monetary"].mean()

segment_summary = (
    customers
    .groupby("Segment")
    .agg(
        Customers=("Customer ID", "count"),
        Revenue=("Monetary", "sum"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
    )
    .reset_index()
)

segment_summary["Customer Share"] = (
    segment_summary["Customers"] / total_customers * 100
)

segment_summary["Revenue Share"] = (
    segment_summary["Revenue"] / total_revenue * 100
)

# Business recommendations tied to the segment names used in the notebook.
SEGMENT_ACTIONS = {
    "Champions / High-Value": {
        "icon": "🏆",
        "description": "Recent, frequent and high-value customers.",
        "action": "Retain and reward with loyalty benefits, VIP treatment and personalized offers.",
    },
    "At-Risk / Mid-Value": {
        "icon": "⚠️",
        "description": "Customers with meaningful value but weaker recent engagement.",
        "action": "Use targeted win-back campaigns, reminders and time-limited incentives.",
    },
    "Recent / Emerging": {
        "icon": "🌱",
        "description": "Recently active customers whose relationship is still developing.",
        "action": "Nurture with onboarding, cross-sell recommendations and repeat-purchase incentives.",
    },
    "Inactive / Low-Value": {
        "icon": "💤",
        "description": "Low-engagement customers contributing relatively little revenue.",
        "action": "Use low-cost reactivation campaigns and avoid over-investing in expensive retention activity.",
    },
}


def money(value):
    if abs(value) >= 1_000_000:
        return f"£{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"£{value / 1_000:.1f}K"
    return f"£{value:,.0f}"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9ca3af;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    .insight-card {
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 12px;
        padding: 16px;
        min-height: 145px;
        background: rgba(128,128,128,0.04);
    }

    .insight-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .insight-value {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .insight-text {
        color: #9ca3af;
        font-size: 0.9rem;
    }

    .recommendation {
        border-left: 4px solid #60a5fa;
        padding: 12px 16px;
        border-radius: 6px;
        background: rgba(96,165,250,0.08);
        margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Customer Segmentation")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Customer Explorer",
        "Segment Analysis",
        "Model Validation",
        "Data Explorer",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("RFM + K-Means Customer Segmentation")


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<p class="main-title">Customer Segmentation Dashboard</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="subtitle">RFM-based customer behavior, cluster structure and business actions</p>',
        unsafe_allow_html=True,
    )

    # KPI row
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers", f"{total_customers:,}")
    c2.metric("Total Revenue", money(total_revenue))
    c3.metric("Avg Customer Value", money(avg_customer_value))
    c4.metric("Segments", f"{customers['Segment'].nunique()}")

    st.markdown("---")

    # --------------------------------------------------------
    # KEY INSIGHTS
    # --------------------------------------------------------

    st.subheader("💡 Key Insights")

    top_revenue_segment = segment_summary.loc[
        segment_summary["Revenue"].idxmax()
    ]

    largest_segment = segment_summary.loc[
        segment_summary["Customers"].idxmax()
    ]

    champions = segment_summary[
        segment_summary["Segment"] == "Champions / High-Value"
    ]

    if not champions.empty:
        champions_row = champions.iloc[0]
        champion_revenue_share = champions_row["Revenue Share"]
        champion_customer_share = champions_row["Customer Share"]
        champion_revenue = champions_row["Revenue"]
    else:
        champion_revenue_share = top_revenue_segment["Revenue Share"]
        champion_customer_share = top_revenue_segment["Customer Share"]
        champion_revenue = top_revenue_segment["Revenue"]

    inactive = segment_summary[
        segment_summary["Segment"] == "Inactive / Low-Value"
    ]

    inactive_revenue_share = (
        inactive.iloc[0]["Revenue Share"]
        if not inactive.empty
        else None
    )

    i1, i2, i3 = st.columns(3)

    with i1:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">🏆 Revenue Concentration</div>
                <div class="insight-value">{champion_revenue_share:.1f}%</div>
                <div class="insight-text">
                    of revenue comes from Champions / High-Value customers,
                    who represent {champion_customer_share:.1f}% of customers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with i2:
        inactive_text = (
            f"Inactive / Low-Value customers contribute only "
            f"{inactive_revenue_share:.1f}% of revenue."
            if inactive_revenue_share is not None
            else f"{largest_segment['Segment']} is the largest customer group."
        )

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">💤 Low-Value Base</div>
                <div class="insight-value">{largest_segment['Customers']:,}</div>
                <div class="insight-text">{inactive_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with i3:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">💰 Top Segment Revenue</div>
                <div class="insight-value">{money(top_revenue_segment['Revenue'])}</div>
                <div class="insight-text">
                    generated by {top_revenue_segment['Segment']}.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --------------------------------------------------------
    # REVENUE + CUSTOMER COUNT
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:
        revenue_chart = segment_summary.sort_values(
            "Revenue",
            ascending=False,
        )

        fig = px.bar(
            revenue_chart,
            x="Segment",
            y="Revenue",
            title="Revenue by Segment",
            text="Revenue",
        )

        fig.update_traces(
            texttemplate="£%{y:.3s}",
            textposition="outside",
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Revenue",
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        customer_chart = segment_summary.sort_values(
            "Customers",
            ascending=False,
        )

        fig = px.bar(
            customer_chart,
            x="Segment",
            y="Customers",
            title="Customers by Segment",
            text="Customers",
        )

        fig.update_traces(
            textposition="outside",
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Customers",
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # RFM SEGMENT VISUALIZATION
    # --------------------------------------------------------

    st.subheader("📈 RFM Segment Positioning")

    rfm_plot = (
        customers
        .groupby("Segment")
        .agg(
            Recency=("Recency", "median"),
            Frequency=("Frequency", "median"),
            Monetary=("Monetary", "median"),
            Customers=("Customer ID", "count"),
        )
        .reset_index()
    )

    fig = px.scatter(
        rfm_plot,
        x="Frequency",
        y="Monetary",
        size="Customers",
        color="Segment",
        hover_data=["Recency", "Customers"],
        text="Segment",
        title="Median Frequency vs Median Monetary Value",
    )

    fig.update_traces(
        textposition="top center",
    )

    fig.update_layout(
        xaxis_title="Median Purchase Frequency",
        yaxis_title="Median Monetary Value",
        legend_title="Segment",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Bubble size represents customer count. "
        "Lower recency values indicate more recent activity."
    )

    # --------------------------------------------------------
    # SEGMENT TABLE
    # --------------------------------------------------------

    st.subheader("Segment Overview")

    display_summary = segment_summary.copy()

    display_summary["Revenue"] = display_summary["Revenue"].map(money)
    display_summary["Avg_Monetary"] = display_summary["Avg_Monetary"].map(money)
    display_summary["Customer Share"] = display_summary["Customer Share"].map(
        lambda x: f"{x:.1f}%"
    )
    display_summary["Revenue Share"] = display_summary["Revenue Share"].map(
        lambda x: f"{x:.1f}%"
    )

    st.dataframe(
        display_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Business takeaway: prioritize retention of high-value customers, "
        "target declining customers with win-back actions, and keep "
        "reactivation costs low for low-value inactive customers."
    )


# ============================================================
# CUSTOMER EXPLORER
# ============================================================

elif page == "Customer Explorer":

    st.title("🔎 Customer Explorer")

    customer_ids = sorted(
        customers["Customer ID"].dropna().unique()
    )

    selected_customer = st.selectbox(
        "Select Customer ID",
        customer_ids,
    )

    customer = customers[
        customers["Customer ID"] == selected_customer
    ].iloc[0]

    st.subheader(f"Customer {selected_customer}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Segment", customer["Segment"])
    c2.metric("Recency", f"{customer['Recency']:.0f} days")
    c3.metric("Frequency", f"{customer['Frequency']:.0f}")
    c4.metric("Monetary", money(customer["Monetary"]))

    if "AOV" in customers.columns:
        st.metric(
            "Average Order Value",
            money(customer["AOV"]),
        )

    # Customer recommendation
    segment_info = SEGMENT_ACTIONS.get(
        customer["Segment"],
        {
            "icon": "📌",
            "description": "Customer segment identified by the clustering model.",
            "action": "Use the RFM profile to determine the appropriate customer strategy.",
        },
    )

    st.markdown(
        f"""
        <div class="recommendation">
            <strong>{segment_info['icon']} Recommended Action</strong><br>
            {segment_info['action']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.subheader("Customer vs Segment Median")

    segment_customers = customers[
        customers["Segment"] == customer["Segment"]
    ]

    comparison = pd.DataFrame(
        {
            "Metric": ["Recency", "Frequency", "Monetary"],
            "Customer": [
                customer["Recency"],
                customer["Frequency"],
                customer["Monetary"],
            ],
            "Segment Median": [
                segment_customers["Recency"].median(),
                segment_customers["Frequency"].median(),
                segment_customers["Monetary"].median(),
            ],
        }
    )

    fig = px.bar(
        comparison,
        x="Metric",
        y=["Customer", "Segment Median"],
        barmode="group",
        title="Customer RFM Profile vs Segment Median",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Customer Details")

    details = pd.DataFrame(
        {
            "Metric": [
                "Customer ID",
                "Segment",
                "Cluster",
                "Recency",
                "Frequency",
                "Monetary",
                "AOV",
            ],
            "Value": [
                customer.get("Customer ID"),
                customer.get("Segment"),
                customer.get("Cluster"),
                customer.get("Recency"),
                customer.get("Frequency"),
                customer.get("Monetary"),
                customer.get("AOV", None),
            ],
        }
    )

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SEGMENT ANALYSIS
# ============================================================

elif page == "Segment Analysis":

    st.title("🎯 Segment Analysis")

    segments = sorted(
        customers["Segment"].dropna().unique()
    )

    selected_segment = st.selectbox(
        "Select Segment",
        segments,
    )

    segment_df = customers[
        customers["Segment"] == selected_segment
    ].copy()

    segment_row = segment_summary[
        segment_summary["Segment"] == selected_segment
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers", f"{len(segment_df):,}")
    c2.metric(
        "Customer Share",
        f"{segment_row['Customer Share']:.1f}%",
    )
    c3.metric(
        "Revenue",
        money(segment_row["Revenue"]),
    )
    c4.metric(
        "Revenue Share",
        f"{segment_row['Revenue Share']:.1f}%",
    )

    info = SEGMENT_ACTIONS.get(
        selected_segment,
        {
            "icon": "📌",
            "description": "Segment identified by the clustering model.",
            "action": "Use the segment's RFM profile to determine the customer strategy.",
        },
    )

    st.markdown(
        f"""
        <div class="recommendation">
            <strong>{info['icon']} {selected_segment}</strong><br>
            {info['description']}<br><br>
            <strong>Recommended action:</strong> {info['action']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            segment_df,
            x="Monetary",
            nbins=40,
            title="Monetary Distribution",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        fig = px.scatter(
            segment_df,
            x="Frequency",
            y="Monetary",
            hover_data=["Customer ID", "Recency"],
            title="Frequency vs Monetary",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader("RFM Profile")

    profile = pd.DataFrame(
        {
            "Metric": ["Recency", "Frequency", "Monetary"],
            "Segment Median": [
                segment_df["Recency"].median(),
                segment_df["Frequency"].median(),
                segment_df["Monetary"].median(),
            ],
            "Overall Median": [
                customers["Recency"].median(),
                customers["Frequency"].median(),
                customers["Monetary"].median(),
            ],
        }
    )

    fig = px.bar(
        profile,
        x="Metric",
        y=["Segment Median", "Overall Median"],
        barmode="group",
        title="Segment vs Overall Median",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Segment Customers")

    columns = [
        col
        for col in [
            "Customer ID",
            "Recency",
            "Frequency",
            "Monetary",
            "AOV",
            "Cluster",
            "Segment",
        ]
        if col in segment_df.columns
    ]

    st.dataframe(
        segment_df[columns].sort_values(
            "Monetary",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv = segment_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Segment CSV",
        data=csv,
        file_name=(
            f"{selected_segment.replace('/', '_').replace(' ', '_')}.csv"
        ),
        mime="text/csv",
    )


# ============================================================
# MODEL VALIDATION
# ============================================================

elif page == "Model Validation":

    st.title("🧪 Model Validation")

    st.markdown(
        "This page exposes the evidence used to select and validate the clustering solution."
    )

    if model_results.empty:

        st.warning(
            "No model_results table was found or it is empty."
        )

        st.markdown(
            """
            Add the validation results from the notebook to the
            `model_results` table, including the K=2 to K=10 model
            comparison and ARI stability results.
            """
        )

    else:

        # Current / latest model
        latest = model_results.iloc[-1]

        st.subheader("Selected Model")

        metric_keys = [
            ("Optimal K", "K"),
            ("Silhouette", "Silhouette"),
            ("Calinski-Harabasz", "Calinski_Harabasz"),
            ("Davies-Bouldin", "Davies_Bouldin"),
        ]

        cols = st.columns(4)

        for col, (label, key) in zip(
            cols,
            metric_keys,
        ):
            if key in latest.index:
                value = latest[key]

                if label == "Optimal K":
                    col.metric(label, f"{int(value)}")
                else:
                    col.metric(label, f"{float(value):.4f}")

        st.markdown("---")

        # K sweep if the table contains multiple K values.
        if len(model_results) > 1 and "K" in model_results.columns:

            st.subheader("K Selection Comparison")

            k_results = model_results.copy()

            required = {
                "K",
                "Silhouette",
                "Calinski_Harabasz",
                "Davies_Bouldin",
            }

            if required.issubset(k_results.columns):

                left, right = st.columns(2)

                with left:
                    fig = px.line(
                        k_results,
                        x="K",
                        y="Silhouette",
                        markers=True,
                        title="Silhouette Score by K",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                with right:
                    fig = px.line(
                        k_results,
                        x="K",
                        y="Davies_Bouldin",
                        markers=True,
                        title="Davies-Bouldin Score by K",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                fig = px.line(
                    k_results,
                    x="K",
                    y="Calinski_Harabasz",
                    markers=True,
                    title="Calinski-Harabasz Score by K",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.subheader("Validation Table")

                st.dataframe(
                    k_results,
                    use_container_width=True,
                    hide_index=True,
                )

        else:
            st.info(
                "The database currently contains one model result. "
                "For the full K=2–10 validation dashboard, store each K's "
                "Silhouette, Calinski-Harabasz and Davies-Bouldin scores "
                "in model_results."
            )

        # Stability
        st.subheader("Cluster Stability")

        stability_cols = [
            col
            for col in [
                "Mean_ARI",
                "Min_ARI",
                "Max_ARI",
            ]
            if col in latest.index
        ]

        if stability_cols:

            c1, c2, c3 = st.columns(3)

            if "Mean_ARI" in latest.index:
                c1.metric(
                    "Mean ARI",
                    f"{float(latest['Mean_ARI']):.4f}",
                )

            if "Min_ARI" in latest.index:
                c2.metric(
                    "Minimum ARI",
                    f"{float(latest['Min_ARI']):.4f}",
                )

            if "Max_ARI" in latest.index:
                c3.metric(
                    "Maximum ARI",
                    f"{float(latest['Max_ARI']):.4f}",
                )

        st.info(
            "Interpretation: higher Silhouette and Calinski-Harabasz values "
            "are generally better, while lower Davies-Bouldin values are better. "
            "ARI is used to assess whether cluster assignments remain stable "
            "across repeated resampling runs."
        )

        st.subheader("Stored Model Results")

        st.dataframe(
            model_results,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.title("🗄️ Data Explorer")

    table = st.selectbox(
        "Select database table",
        [
            "customers",
            "transactions",
            "model_results",
        ],
    )

    df = load_table(table)

    st.write(f"Rows: {len(df):,}")
    st.write(f"Columns: {len(df.columns):,}")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Table CSV",
        data=csv,
        file_name=f"{table}.csv",
        mime="text/csv",
    )
