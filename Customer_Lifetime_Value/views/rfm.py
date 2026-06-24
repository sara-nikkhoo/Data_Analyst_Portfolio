import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os, sys
# Resolve artifacts path relative to app.py location
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ARTIFACTS_DIR = os.path.join(_ROOT, "artifacts")

SEGMENT_COLORS = {
    "Champions":          "#38a169",
    "Loyal Customers":    "#3182ce",
    "Potential Loyalists":"#0bc5ea",
    "New Customers":      "#68d391",
    "Need Attention":     "#d69e2e",
    "At Risk":            "#ed8936",
    "Cannot Lose Them":   "#e53e3e",
    "Hibernating":        "#a0aec0",
}

SEGMENT_ACTIONS = {
    "Champions":           "VIP rewards · early product access · referral programme",
    "Loyal Customers":     "Loyalty points · upsell complementary categories · birthday offer",
    "Potential Loyalists": "2nd-purchase incentive (10% off) · personalised recommendations",
    "New Customers":       "Welcome sequence · category discovery · free shipping on 2nd order",
    "Need Attention":      "Seasonal offer aligned to last purchase category",
    "At Risk":             "Win-back campaign — personalised 15% discount on past categories",
    "Cannot Lose Them":    "High-touch re-engagement — 20% discount + departure survey",
    "Hibernating":         "Last-chance email — no response → suppress",
}


@st.cache_resource
def load_rfm():
    try:
        rfm  = joblib.load(f"{ARTIFACTS_DIR}/rfm.pkl")
        meta = joblib.load(f"{ARTIFACTS_DIR}/model_meta.pkl")
        return rfm, meta
    except FileNotFoundError:
        return None, None


def card(label, value, sub="", style=""):
    return (
        f"<div class='card {style}'>"
        f"<div class='card-label'>{label}</div>"
        f"<div class='card-value'>{value}</div>"
        f"{'<div class=card-sub>' + sub + '</div>' if sub else ''}"
        f"</div>"
    )


def show():
    rfm, meta = load_rfm()

    if rfm is None:
        st.error("❌ Artifacts not found. Run `python train_pipeline.py` first.")
        return

    st.markdown("##   RFM Segmentation")
    st.markdown(
        f"<div class='info-box'>Dataset snapshot: <b>{meta['snapshot']}</b> · "
        f"{meta['n_customers']:,} customers · "
        f"£{meta['total_revenue']:,.0f} total revenue · "
        f"Churn rate: {meta['churn_rate']*100:.1f}% (data-driven definition)</div>",
        unsafe_allow_html=True
    )

    # ── Sidebar filters
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.8rem; font-weight:600; color:#a0aec0;'>RFM FILTERS</div>",
            unsafe_allow_html=True
        )
        seg_filter = st.multiselect(
            "Segments",
            options=sorted(rfm["segment"].unique()),
            default=sorted(rfm["segment"].unique()),
            key="rfm_seg"
        )
        top_n = st.number_input("Top N customers", value=100, min_value=10, key="rfm_topn")
        sort_by = st.selectbox(
            "Sort table by",
            ["rfm_score", "total_spend", "recency", "frequency", "monetary"],
            key="rfm_sort"
        )

    filtered = rfm[rfm["segment"].isin(seg_filter)]

    # ── KPI Row
    champions     = len(rfm[rfm["segment"] == "Champions"])
    at_risk       = len(rfm[rfm["segment"] == "At Risk"])
    cant_lose     = len(rfm[rfm["segment"] == "Cannot Lose Them"])
    rev_at_risk   = rfm[rfm["segment"].isin(["At Risk","Cannot Lose Them"])]["total_spend"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(card("Total Customers", f"{len(rfm):,}", f"{meta['n_segments']} segments"), unsafe_allow_html=True)
    c2.markdown(card("Champions", f"{champions:,}", f"{champions/len(rfm)*100:.1f}% of base", "success"), unsafe_allow_html=True)
    c3.markdown(card("At Risk", f"{at_risk:,}", "Need win-back", "warning"), unsafe_allow_html=True)
    c4.markdown(card("Cannot Lose Them", f"{cant_lose:,}", "High value, lapsed", "danger"), unsafe_allow_html=True)
    c5.markdown(card("Revenue at Risk", f"£{rev_at_risk:,.0f}", "At Risk + Cannot Lose", "danger"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("<div class='section-title'>Customer Distribution by Segment</div>", unsafe_allow_html=True)
        seg_counts = rfm.groupby("segment").size().reset_index(name="customers")
        seg_counts = seg_counts.sort_values("customers", ascending=True)
        
        fig = go.Figure(go.Bar(
            y=seg_counts["segment"],
            x=seg_counts["customers"],
            orientation="h",
            marker=dict(
                color=seg_counts["customers"],
                colorscale=[[0, "#e8f5e9"], [1, "#2b9348"]],
                showscale=False
            ),
            text=[f"{v} ({v/seg_counts['customers'].sum()*100:.1f}%)" for v in seg_counts["customers"]],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            height=320, margin=dict(t=10, b=10, l=150, r=80),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-title'>Revenue Share by Segment</div>", unsafe_allow_html=True)
        seg_rev = (
            rfm.groupby("segment")["total_spend"]
            .sum().reset_index()
            .sort_values("total_spend", ascending=True)
        )
        seg_rev["pct"] = (seg_rev["total_spend"] / seg_rev["total_spend"].sum() * 100).round(1)
        fig2 = go.Figure(go.Bar(
            y=seg_rev["segment"],
            x=seg_rev["total_spend"],
            orientation="h",
            marker=dict(
                color=seg_rev["total_spend"],
                colorscale=[[0, "#f5d5dc"], [1, "#6a040f"]],
                showscale=False
            ),
            text=[f"£{v:,.0f}  ({p}%)" for v, p in zip(seg_rev["total_spend"], seg_rev["pct"])],
            textposition="outside",
        ))
        fig2.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            height=320, margin=dict(t=10, b=10, l=10, r=120),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── RFM Distribution
    st.markdown("<div class='section-title'>RFM Feature Distributions</div>", unsafe_allow_html=True)
    fig3 = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Recency (days)", "Frequency (orders)", "Monetary (avg order £)"]
    )
    for i, (col, color) in enumerate(
        zip(["recency", "frequency", "monetary"], ["#6a040f", "#3182ce", "#38a169"]), 1
    ):
        fig3.add_trace(
            go.Histogram(x=rfm[col], nbinsx=50, marker_color=color,
                         opacity=0.75, name=col),
            row=1, col=i
        )
        fig3.add_vline(
            x=rfm[col].median(), line_dash="dash", line_color="#1a202c",
            row=1, col=i
        )
    fig3.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        showlegend=False, height=250,
        margin=dict(t=30, b=20, l=20, r=20)
    )
    fig3.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig3.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Segment profiles + actions
    st.markdown("<div class='section-title'>Segment Profiles & Recommended Actions</div>", unsafe_allow_html=True)

    seg_profile = (
        rfm.groupby("segment")
        .agg(
            customers    =("Customer ID",  "count"),
            avg_recency  =("recency",      "mean"),
            avg_frequency=("frequency",    "mean"),
            avg_monetary =("monetary",     "mean"),
            total_revenue=("total_spend",  "sum"),
            avg_rfm      =("rfm_score",    "mean"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    seg_profile["revenue_%"] = (seg_profile["total_revenue"] / seg_profile["total_revenue"].sum() * 100).round(1)

    for _, row in seg_profile.iterrows():
        color  = SEGMENT_COLORS.get(row["segment"], "#a0aec0")
        action = SEGMENT_ACTIONS.get(row["segment"], "")
        with st.expander(
            f"**{row['segment']}** — "
            f"{int(row['customers']):,} customers · "
            f"£{row['total_revenue']:,.0f} ({row['revenue_%']}% of revenue)"
        ):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Recency",   f"{row['avg_recency']:.0f}d")
            m2.metric("Avg Frequency", f"{row['avg_frequency']:.1f}")
            m3.metric("Avg Order",     f"£{row['avg_monetary']:.0f}")
            m4.metric("Avg RFM Score", f"{row['avg_rfm']:.1f}")
            st.markdown(
                f"<div style='background:#f7fafc; border-left:4px solid {color}; "
                f"padding:0.6rem 1rem; border-radius:0 6px 6px 0; "
                f"font-size:0.82rem; color:#2d3748; margin-top:0.5rem;'>"
                f"<b>Action:</b> {action}</div>",
                unsafe_allow_html=True
            )

    # ── Customer table
    st.markdown("<div class='section-title'>Customer List</div>", unsafe_allow_html=True)

    display_cols = ["Customer ID", "segment", "recency", "frequency",
                    "monetary", "total_spend", "rfm_score", "km_label"]
    ######


    
    table = (
        filtered
        .sort_values(sort_by, ascending=False)
        .head(int(top_n))[display_cols]
        .reset_index(drop=True)
    )

    # Round float columns
    float_cols = table.select_dtypes(include="float").columns
    table[float_cols] = table[float_cols].round(2)

    def color_segment(val):
        colors_map = {
            "Champions":          "color:#276749; font-weight:600;",
            "Loyal Customers":    "color:#2c5282; font-weight:600;",
            "Cannot Lose Them":   "color:#9b2c2c; font-weight:600;",
            "At Risk":            "color:#7b341e; font-weight:600;",
            "Hibernating":        "color:#718096;",
            "Potential Loyalists":"color:#086f83;",
            "New Customers":      "color:#22543d;",
            "Need Attention":     "color:#744210;",
        }
        return colors_map.get(val, "")

    styled = table.style.map(color_segment, subset=["segment"])

    st.dataframe(styled, use_container_width=True, height=400)

    # ── Export
    csv = filtered[display_cols].to_csv(index=False)
    st.download_button(
        "⬇️  Export Segment Data (CSV)",
        csv, "rfm_segments.csv", "text/csv"
    )