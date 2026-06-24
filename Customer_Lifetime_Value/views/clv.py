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


def _rebuild_bgnbd(params_dict: dict):
    """Reconstruct BetaGeoFitter from saved parameters."""
    from lifetimes import BetaGeoFitter
    import numpy as np
    bgf = BetaGeoFitter(penalizer_coef=params_dict["penalizer_coef"])
    # Manually set fitted parameters — bypasses joblib serialization issue
    bgf.params_ = {k: np.float64(v) for k, v in params_dict["params"].items()}
    bgf._fitted_parameters_names = list(params_dict["params"].keys())
    return bgf


def _rebuild_ggf(params_dict: dict):
    """Reconstruct GammaGammaFitter from saved parameters."""
    from lifetimes import GammaGammaFitter
    import numpy as np
    ggf = GammaGammaFitter(penalizer_coef=params_dict["penalizer_coef"])
    ggf.params_ = {k: np.float64(v) for k, v in params_dict["params"].items()}
    ggf._fitted_parameters_names = list(params_dict["params"].keys())
    return ggf


@st.cache_resource
def load_artifacts():
    try:
        rfm          = joblib.load(f"{ARTIFACTS_DIR}/rfm.pkl")
        meta         = joblib.load(f"{ARTIFACTS_DIR}/model_meta.pkl")
        top_products = joblib.load(f"{ARTIFACTS_DIR}/top_products.pkl")
        return rfm, meta, top_products
    except FileNotFoundError:
        return None, None, None


def card(label, value, sub="", style=""):
    return (
        f"<div class='card {style}'>"
        f"<div class='card-label'>{label}</div>"
        f"<div class='card-value'>{value}</div>"
        f"{'<div class=card-sub>' + sub + '</div>' if sub else ''}"
        f"</div>"
    )


def show():
    rfm, meta, top_products = load_artifacts()

    if rfm is None:
        st.error("❌ Artifacts not found. Run `python train_pipeline.py` first.")
        return

    st.markdown("##  CLV Prediction")

    # ── Model selector (sidebar)
    with st.sidebar:
        st.markdown(
            "<div style='font-size:0.8rem; font-weight:600; color:#a0aec0;'>CLV MODEL</div>",
            unsafe_allow_html=True
        )
        model_choice = st.radio(
            "Select model",
            ["BG/NBD + Gamma-Gamma", "ML (XGBoost Tweedie)"],
            key="clv_model"
        )
        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.8rem; font-weight:600; color:#a0aec0;'>FILTERS</div>",
            unsafe_allow_html=True
        )
        seg_filter = st.multiselect(
            "Segments",
            options=sorted(rfm["segment"].unique()),
            default=sorted(rfm["segment"].unique()),
            key="clv_seg"
        )
        top_n = st.number_input("Top N customers", value=100, min_value=10, key="clv_topn")

        # CLV tiers
        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.8rem; font-weight:600; color:#a0aec0;'>CLV TIERS</div>",
            unsafe_allow_html=True
        )

    # ── Determine active CLV column based on model choice
    is_bgnbd = "BG/NBD" in model_choice
    clv_col  = "clv_12m"   if is_bgnbd else "pred_clv_90d"
    horizon  = "12 months" if is_bgnbd else "90 days"

    # Check column exists
    if clv_col not in rfm.columns:
        missing_model = "BG/NBD" if is_bgnbd else "ML XGBoost"
        st.markdown(
            f"<div class='warn-box'>⚠️  {missing_model} predictions not found in artifacts. "
            f"Re-run <code>train_pipeline.py</code>.</div>",
            unsafe_allow_html=True
        )
        return

    # Filter
    filtered = rfm[rfm["segment"].isin(seg_filter)].copy()
    filtered = filtered.dropna(subset=[clv_col])

    # ── Model info banner
    badge_class = "badge-bgnbd" if is_bgnbd else "badge-ml"
    badge_label = "BG/NBD + Gamma-Gamma" if is_bgnbd else "XGBoost Tweedie"
    st.markdown(
        f"<div class='info-box'>"
        f"Active model: <span class='model-badge {badge_class}'>{badge_label}</span> · "
        f"Prediction horizon: <b>{horizon}</b> · "
        f"Snapshot: <b>{meta['snapshot']}</b>"
        f"{'  |  CV R²: ' + str(meta['ml_metrics']['reg_cv_r2']) if not is_bgnbd else ''}"
        f"{'  |  AUC: ' + str(meta['ml_metrics']['clf_auc']) if not is_bgnbd else ''}"
        f"</div>",
        unsafe_allow_html=True
    )

    # Model explanation
    with st.expander("ℹ️  Model explanation & assumptions"):
        if is_bgnbd:
            st.markdown("""
            **BG/NBD + Gamma-Gamma (Probabilistic Model)**

            Two components:
            - **BG/NBD**: models how often a customer buys while still active, and when they permanently churn
            - **Gamma-Gamma**: models expected spend per transaction given repeat purchase history

            **Output**: Expected revenue over 12 months, discounted at 1%/month (~12% annual).
            Only customers with ≥1 repeat purchase receive a Gamma-Gamma estimate.

            **Assumptions**: purchase rate and dropout probability are heterogeneous across customers
            (Gamma and Beta distributions respectively). Spend is independent of purchase frequency.

            **Profit margin applied**: {:.0f}%
            """.format(meta["profit_margin"] * 100))
        else:
            m = meta["ml_metrics"]
            st.markdown(f"""
            **XGBoost Tweedie Regression + Purchase Classifier (ML Model)**

            Two components trained on a **time-based split**:
            - **Regression** (Tweedie loss): predicts spend in next 90 days from past behaviour
            - **Classifier**: predicts probability of making any purchase in next 90 days

            **Combined score**: E[CLV] = P(purchase) × predicted spend

            **Why Tweedie loss**: spend is zero-inflated and right-skewed.
            Standard MSE assumes normal errors — Tweedie is designed for this distribution.

            **Validation** (out-of-sample test set):
            - Regression CV R² = {m["reg_cv_r2"]} · Test R² = {m["reg_test_r2"]} · MAE = £{m["reg_mae"]}
            - Classifier CV AUC = {m["clf_cv_auc"]} · Test AUC = {m["clf_auc"]} · F1 = {m["clf_f1"]}
            - Optimal threshold = {m["threshold"]} (derived from ROC curve, not default 0.5)

            **Features**: recency, frequency, historical spend, rolling 28d + 14d windows,
            product diversity (n_unique_products, n_unique_descriptions)
            """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CLV Tier assignment
    p33 = filtered[clv_col].quantile(0.33)
    p66 = filtered[clv_col].quantile(0.66)

    def assign_tier(v):
        if v >= p66: return "High CLV"
        if v >= p33: return "Mid CLV"
        return "Low CLV"

    filtered["clv_tier"] = filtered[clv_col].apply(assign_tier)

    # Sidebar tier filter
    with st.sidebar:
        tier_filter = st.multiselect(
            "CLV Tiers",
            ["High CLV", "Mid CLV", "Low CLV"],
            default=["High CLV", "Mid CLV", "Low CLV"],
            key="clv_tier"
        )

    filtered = filtered[filtered["clv_tier"].isin(tier_filter)]

    # ── KPI Row
    high_clv  = filtered[filtered["clv_tier"] == "High CLV"]
    mid_clv   = filtered[filtered["clv_tier"] == "Mid CLV"]
    low_clv   = filtered[filtered["clv_tier"] == "Low CLV"]
    total_clv = filtered[clv_col].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(card("High CLV",  f"{len(high_clv):,}",
                     f"≥ £{p66:.0f}", "success"), unsafe_allow_html=True)
    c2.markdown(card("Mid CLV",   f"{len(mid_clv):,}",
                     f"£{p33:.0f} – £{p66:.0f}", "warning"), unsafe_allow_html=True)
    c3.markdown(card("Low CLV",   f"{len(low_clv):,}",
                     f"< £{p33:.0f}"), unsafe_allow_html=True)
    c4.markdown(card("Total Predicted",  f"£{total_clv:,.0f}",
                     horizon, "purple"), unsafe_allow_html=True)
    if is_bgnbd and "prob_alive" in filtered.columns:
        avg_alive = filtered["prob_alive"].mean()
        c5.markdown(card("Avg P(Active)",   f"{avg_alive*100:.0f}%",
                         "BG/NBD estimate", "teal"), unsafe_allow_html=True)
    elif not is_bgnbd and "pred_prob" in filtered.columns:
        avg_prob = filtered["pred_prob"].mean()
        c5.markdown(card("Avg P(Purchase)", f"{avg_prob*100:.0f}%",
                         "90-day probability", "teal"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("<div class='section-title'>CLV Distribution</div>",
                    unsafe_allow_html=True)
        fig = go.Figure()
        tier_colors = {"High CLV": "#38a169", "Mid CLV": "#d69e2e", "Low CLV": "#e53e3e"}
        for tier in ["Low CLV", "Mid CLV", "High CLV"]:
            subset = filtered[filtered["clv_tier"] == tier][clv_col]
            if len(subset):
                fig.add_trace(go.Histogram(
                    x=subset, nbinsx=40, name=tier,
                    marker_color=tier_colors[tier], opacity=0.75
                ))
        fig.add_vline(x=filtered[clv_col].median(), line_dash="dash",
                      line_color="#1a202c", annotation_text="Median")
        fig.update_layout(
            barmode="overlay",
            paper_bgcolor="white", plot_bgcolor="white",
            height=300, margin=dict(t=10, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(title=f"Predicted CLV (£) — {horizon}",
                       showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(title="Customers", showgrid=True, gridcolor="#f0f0f0")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-title'>Avg CLV by Segment</div>",
                    unsafe_allow_html=True)
        seg_clv = (
            filtered.groupby("segment")[clv_col]
            .mean().reset_index()
            .sort_values(clv_col, ascending=True)
        )
        fig2 = go.Figure(go.Bar(
            y=seg_clv["segment"],
            x=seg_clv[clv_col],
            orientation="h",
            marker=dict(
                color=seg_clv[clv_col],
                colorscale=[[0, "#f5d5dc"], [1, "#6a040f"]],
                showscale=False
            ),
            text=[f"£{v:,.0f}" for v in seg_clv[clv_col]],
            textposition="outside",
        ))
        fig2.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            height=300, margin=dict(t=10, b=20, l=10, r=80),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Second chart row: CLV tier pie + BG/NBD alive / ML prob scatter
    col_l2, col_r2 = st.columns([1, 1])

    with col_l2:
        st.markdown("<div class='section-title'>CLV Tier Split</div>",
                    unsafe_allow_html=True)
        tier_counts = filtered["clv_tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        fig3 = px.pie(
            tier_counts, values="count", names="tier",
            color="tier",
            color_discrete_map={"High CLV":"#fb8b24","Mid CLV":"#0f4c5c","Low CLV":"#5f0f40"},
            hole=0.4
        )
        fig3.update_traces(
            textposition="auto",
            textinfo="label+percent",
            textfont=dict(size=12)
        )
        fig3.update_layout(
            paper_bgcolor="white", showlegend=True,
            height=350, margin=dict(t=10, b=10, l=10, r=150),
            legend=dict(x=1.05, y=1, xanchor='left', yanchor='top')
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        if is_bgnbd and "prob_alive" in filtered.columns:
            st.markdown("<div class='section-title'>Predicted Purchases vs P(Active)</div>",
                        unsafe_allow_html=True)
            sample = filtered.dropna(subset=["predicted_purchases_90d","prob_alive"])
            sample = sample.sample(min(1000, len(sample)), random_state=42)
            fig4 = px.scatter(
                sample,
                x="prob_alive", y="predicted_purchases_90d",
                color="segment", color_discrete_map=SEGMENT_COLORS,
                opacity=0.6, size_max=8,
                labels={"prob_alive": "P(Still Active)",
                        "predicted_purchases_90d": "Expected Purchases (90d)"},
            )
            fig4.update_layout(
                paper_bgcolor="white", plot_bgcolor="white",
                height=350, margin=dict(t=10, b=20, l=20, r=20),
                legend=dict(font=dict(size=9))
            )
            fig4.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
            fig4.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig4, use_container_width=True)

        elif not is_bgnbd and "pred_prob" in filtered.columns and "pred_spend" in filtered.columns:
            st.markdown("<div class='section-title'>P(Purchase) vs Predicted Spend</div>",
                        unsafe_allow_html=True)
            sample = filtered.sample(min(1000, len(filtered)), random_state=42)
            fig4 = px.scatter(
                sample,
                x="pred_prob", y="pred_spend",
                color="segment", color_discrete_map=SEGMENT_COLORS,
                opacity=0.6,
                labels={"pred_prob": "P(Purchase in 90d)",
                        "pred_spend": "Predicted Spend (£)"},
            )
            fig4.update_layout(
                paper_bgcolor="white", plot_bgcolor="white",
                height=350, margin=dict(t=10, b=20, l=20, r=20),
                legend=dict(font=dict(size=9))
            )
            fig4.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
            fig4.update_yaxes(showgrid=True, gridcolor="#f0f0f0", tickprefix="£")
            st.plotly_chart(fig4, use_container_width=True)

    # ── Revenue opportunity
    
    # ── Top products for high-CLV customers
    if top_products is not None and len(top_products):
        st.markdown("<div class='section-title'>Top Products — High CLV Customers</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div class='info-box'>Products most purchased by the top 100 predicted-CLV customers. "
            "Use for targeted retention promotions — these items drove their past engagement.</div>",
            unsafe_allow_html=True
        )
        fig5 = go.Figure(go.Bar(
            y=top_products["Description"][::-1],
            x=top_products["Quantity"][::-1],
            orientation="h",
            marker=dict(
                color=top_products["Quantity"][::-1],
                colorscale=[[0, "#f0e8f0"], [1, "#5f0f40"]],
                showscale=False
            ),
            text=top_products["Quantity"][::-1],
            textposition="outside",
        ))
        fig5.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            height=400, margin=dict(t=10, b=20, l=20, r=60),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            showlegend=False
        )
        st.plotly_chart(fig5, use_container_width=True)

    # ── Customer table
    st.markdown("<div class='section-title'>Customer CLV List</div>",
                unsafe_allow_html=True)

    base_cols = ["Customer ID", "segment", "recency", "frequency",
                 "monetary", "total_spend", "rfm_score"]
    clv_cols  = [clv_col]
    if is_bgnbd:
        for c in ["clv_90d", "clv_12m_profit", "prob_alive", "predicted_purchases_90d"]:
            if c in filtered.columns:
                clv_cols.append(c)
    else:
        for c in ["pred_spend", "pred_prob", "clv_tier"]:
            if c in filtered.columns:
                clv_cols.append(c)

    show_cols = base_cols + [c for c in clv_cols if c not in base_cols]
    show_cols = [c for c in show_cols if c in filtered.columns]

    table = (
        filtered
        .sort_values(clv_col, ascending=False)
        .head(int(top_n))[show_cols]
        .reset_index(drop=True)
    )

    # Rename columns to highlight statistical methods
    if is_bgnbd:
        rename_map = {
            "clv_12m": "clv_12m (BG/NBD)",
            "clv_90d": "clv_90d (BG/NBD)",
            "clv_12m_profit": "clv_12m_profit (BG/NBD)",
            "prob_alive": "prob_alive (BG/NBD)",
            "predicted_purchases_90d": "predicted_purchases_90d (BG/NBD)"
        }
        table = table.rename(columns=rename_map)

    float_cols = table.select_dtypes(include="float").columns
    table[float_cols] = table[float_cols].round(2)

    def color_tier(val):
        if val == "High CLV":   return "color:#276749; font-weight:600;"
        if val == "Mid CLV":    return "color:#744210;"
        if val == "Low CLV":    return "color:#9b2c2c;"
        return ""

    def color_segment(val):
        m = {
            "Champions":        "color:#276749; font-weight:600;",
            "Loyal Customers":  "color:#2c5282; font-weight:600;",
            "Cannot Lose Them": "color:#9b2c2c; font-weight:600;",
            "At Risk":          "color:#7b341e; font-weight:600;",
        }
        return m.get(val, "")

    styled = table.style.map(color_segment, subset=["segment"])
    if "clv_tier" in table.columns:
        styled = styled.map(color_tier, subset=["clv_tier"])

    st.dataframe(styled, use_container_width=True, height=420)

    # ── Export
    csv = filtered[show_cols].to_csv(index=False)
    model_label = "bgnbd" if is_bgnbd else "ml"
    st.download_button(
        f"⬇️  Export CLV Predictions ({model_label.upper()}) CSV",
        csv, f"clv_predictions_{model_label}.csv", "text/csv"
    )