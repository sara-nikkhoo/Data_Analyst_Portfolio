"""
=============================================================
  Churn Early Warning System — Streamlit Dashboard
=============================================================
  Run locally  : streamlit run dashboard.py
  Deploy free  : https://share.streamlit.io
                 (connect GitHub repo, select dashboard.py)
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import shap
import warnings
warnings.filterwarnings("ignore")


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Early Warning System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid #378ADD;
    margin-bottom: 0.5rem;
  }
  .metric-card.danger  { border-left-color: #E24B4A; }
  .metric-card.warning { border-left-color: #EF9F27; }
  .metric-card.success { border-left-color: #639922; }
  .metric-label { font-size: 12px; color: #888; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-value { font-size: 28px; font-weight: 700; color: #1a1a1a; }
  .metric-sub   { font-size: 12px; color: #aaa; margin-top: 2px; }
  .section-header { font-size: 18px; font-weight: 600; color: #1a1a1a; margin: 1.5rem 0 0.75rem; border-bottom: 2px solid #f0f0f0; padding-bottom: 0.4rem; }
  .risk-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .high   { background: #fde8e8; color: #A32D2D; }
  .medium { background: #fef3e0; color: #854F0B; }
  .low    { background: #eaf3de; color: #3B6D11; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA & MODEL (cached so it only runs once)
# =============================================================================
@st.cache_data
def load_and_train(uploaded_file=None):
    """Load data, train model, compute SHAP values. Cached after first run."""

    # ── Load ──────────────────────────────────────────────────────────────────
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Fallback: generate synthetic data that mirrors the Kaggle dataset
        np.random.seed(42)
        n = 3333
        df = pd.DataFrame({
            "Account length":               np.random.randint(1, 243, n),
            "International plan":           np.random.choice(["Yes","No"], n, p=[0.1,0.9]),
            "Voice mail plan":              np.random.choice(["Yes","No"], n, p=[0.27,0.73]),
            "Number vmail messages":        np.random.randint(0, 51, n),
            "Total day minutes":            np.random.normal(180, 54, n).clip(0),
            "Total day calls":              np.random.randint(50, 160, n),
            "Total day charge":             np.random.normal(30, 9, n).clip(0),
            "Total eve minutes":            np.random.normal(200, 50, n).clip(0),
            "Total eve calls":              np.random.randint(50, 150, n),
            "Total eve charge":             np.random.normal(17, 4, n).clip(0),
            "Total night minutes":          np.random.normal(200, 55, n).clip(0),
            "Total night calls":            np.random.randint(50, 175, n),
            "Total night charge":           np.random.normal(9, 2.5, n).clip(0),
            "Total intl minutes":           np.random.normal(10, 2.8, n).clip(0),
            "Total intl calls":             np.random.randint(1, 21, n),
            "Total intl charge":            np.random.normal(2.7, 0.75, n).clip(0),
            "Customer service calls":       np.random.randint(0, 10, n),
        })
        # Simulate correlated churn
        churn_prob = (
            0.15
            + 0.25 * (df["International plan"] == "Yes")
            + 0.03 * df["Customer service calls"]
            + 0.002 * (df["Total day minutes"] - 180)
            - 0.20 * (df["Voice mail plan"] == "Yes")
        ).clip(0.02, 0.95)
        df["Churn"] = (np.random.rand(n) < churn_prob).astype(bool)

    # ── Preprocess ────────────────────────────────────────────────────────────
    df = df.copy()
    
    # Clean numeric columns: remove brackets and convert scientific notation
    for col in df.columns:
        if df[col].dtype == object:
            try:
                # Remove brackets if present (e.g., "[5E-1]" -> "5E-1")
                df[col] = df[col].str.replace(r'[\[\]]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
    for col in ["International plan", "Voice mail plan"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"True":1,"False":0,"Yes":1,"No":0})
    else:
        df["Churn"] = df["Churn"].astype(int)

    for c in ["State","Area code","Phone number"]:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # ── Split ─────────────────────────────────────────────────────────────────
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        use_label_encoder=False, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    importance = pd.DataFrame({
        "feature": X_test.columns,
        "shap_value": np.abs(shap_vals).mean(axis=0)
    }).sort_values("shap_value", ascending=False).reset_index(drop=True)

    # ── Risk table ────────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    df_risk = X_test.copy().reset_index(drop=True)
    df_risk["churn_probability"] = (y_prob * 100).round(1)
    df_risk["actual_churn"]      = y_test.values
    df_risk["risk_tier"] = pd.cut(
        df_risk["churn_probability"],
        bins=[0,40,70,100], labels=["Low","Medium","High"]
    )

    charge_col = "Total day charge"
    if charge_col in df_risk.columns:
        df_risk["monthly_value"] = (df_risk[charge_col] * 3).round(2)
    else:
        df_risk["monthly_value"] = np.random.uniform(20, 120, len(df_risk)).round(2)

    df_risk = df_risk.sort_values("churn_probability", ascending=False)
    df_risk.insert(0, "customer_id",
                   [f"CUST-{str(i).zfill(4)}" for i in range(1, len(df_risk)+1)])

    return df_risk, importance, shap_vals, X_test, auc, model


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("⚡ Churn Early Warning")
    st.markdown("---")
    st.markdown("**Upload your dataset**")
    uploaded = st.file_uploader("CSV file", type=["csv"],
                                help="Must include a 'Churn' column (True/False or Yes/No)")

    st.markdown("---")
    st.markdown("**Risk thresholds**")
    high_thresh   = st.slider("High risk above (%)",   40, 90, 70)
    medium_thresh = st.slider("Medium risk above (%)", 10, high_thresh-5, 40)

    st.markdown("---")
    st.markdown("**Early Warning List**")
    top_n = st.slider("Show top N customers", 10, 100, 50)

    st.markdown("---")
    st.caption("Built with XGBoost + SHAP  |  Portfolio project")


# =============================================================================
# LOAD
# =============================================================================
with st.spinner("Training model and computing SHAP values…"):
    df_risk, importance, shap_vals, X_test, auc, model = load_and_train(uploaded)

# Re-apply sidebar thresholds
df_risk["risk_tier"] = pd.cut(
    df_risk["churn_probability"],
    bins=[0, medium_thresh, high_thresh, 100],
    labels=["Low","Medium","High"]
)

# =============================================================================
# HEADER
# =============================================================================
st.title("⚡ Customer Churn Early Warning System")
st.markdown(
    "AI-powered churn prediction using **XGBoost + SHAP**. "
    "Shows *who* is at risk, *why* they're leaving, and *how much revenue* is at stake."
)
st.markdown("---")


# =============================================================================
# KPI CARDS
# =============================================================================
high_risk  = df_risk[df_risk["risk_tier"] == "High"]
med_risk   = df_risk[df_risk["risk_tier"] == "Medium"]
rev_at_risk = high_risk["monthly_value"].sum()
total_rev   = df_risk["monthly_value"].sum()
rev_pct     = (rev_at_risk / total_rev * 100) if total_rev > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class="metric-card danger">
      <div class="metric-label">High Risk Customers</div>
      <div class="metric-value">{len(high_risk):,}</div>
      <div class="metric-sub">Need immediate action</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card warning">
      <div class="metric-label">Medium Risk</div>
      <div class="metric-value">{len(med_risk):,}</div>
      <div class="metric-sub">Monitor closely</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card danger">
      <div class="metric-label">Revenue at Risk</div>
      <div class="metric-value">${rev_at_risk:,.0f}</div>
      <div class="metric-sub">{rev_pct:.1f}% of total monthly</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Model ROC-AUC</div>
      <div class="metric-value">{auc:.3f}</div>
      <div class="metric-sub">XGBoost accuracy signal</div>
    </div>""", unsafe_allow_html=True)
with c5:
    avg_prob = df_risk[df_risk["risk_tier"]=="High"]["churn_probability"].mean()
    st.markdown(f"""
    <div class="metric-card danger">
      <div class="metric-label">Avg High-Risk Score</div>
      <div class="metric-value">{avg_prob:.0f}%</div>
      <div class="metric-sub">Avg churn probability</div>
    </div>""", unsafe_allow_html=True)


# =============================================================================
# ROW 2: FEATURE IMPORTANCE + RISK DISTRIBUTION
# =============================================================================
st.markdown('<div class="section-header">Churn Drivers & Risk Distribution</div>',
            unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("**Top churn drivers (SHAP feature importance)**")
    top10 = importance.head(10).sort_values("shap_value")
    colors = ["#E24B4A" if i >= 7 else "#378ADD" for i in range(len(top10))]
    fig_imp = go.Figure(go.Bar(
        x=top10["shap_value"],
        y=top10["feature"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig_imp.update_layout(
        xaxis_title="Mean |SHAP Value|",
        height=360, margin=dict(l=10, r=10, t=10, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=12),
    )
    fig_imp.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig_imp.update_yaxes(showgrid=False)
    st.plotly_chart(fig_imp, use_container_width=True)

with col_right:
    st.markdown("**Churn probability distribution**")
    fig_hist = px.histogram(
        df_risk, x="churn_probability",
        color="risk_tier",
        color_discrete_map={"High":"#E24B4A","Medium":"#EF9F27","Low":"#639922"},
        nbins=30, opacity=0.8,
        labels={"churn_probability":"Churn Probability (%)","risk_tier":"Risk Tier"},
        category_orders={"risk_tier":["High","Medium","Low"]},
    )
    fig_hist.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        legend_title_text="Risk Tier",
        bargap=0.05,
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# =============================================================================
# ROW 3: REVENUE AT RISK BREAKDOWN
# =============================================================================
st.markdown('<div class="section-header">Revenue at Risk Breakdown</div>',
            unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1.2, 1, 1])

with col_a:
    tier_rev = df_risk.groupby("risk_tier", observed=True)["monthly_value"].sum()
    tier_rev = tier_rev.reindex(["High","Medium","Low"])
    fig_pie = go.Figure(go.Pie(
        labels=tier_rev.index,
        values=tier_rev.values,
        hole=0.5,
        marker_colors=["#E24B4A","#EF9F27","#639922"],
        textinfo="label+percent",
        hovertemplate="%{label}: $%{value:,.0f}<extra></extra>",
    ))
    fig_pie.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False, paper_bgcolor="white",
    )
    st.markdown("**Monthly revenue by tier**")
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    st.markdown("**Monthly revenue summary**")
    tier_summary = df_risk.groupby("risk_tier", observed=True).agg(
        customers=("customer_id","count"),
        total_revenue=("monthly_value","sum"),
        avg_probability=("churn_probability","mean")
    ).reindex(["High","Medium","Low"])
    tier_summary["total_revenue"] = tier_summary["total_revenue"].map("${:,.0f}".format)
    tier_summary["avg_probability"] = tier_summary["avg_probability"].map("{:.1f}%".format)
    st.dataframe(tier_summary, use_container_width=True)

with col_c:
    st.markdown("**Top churn-driving feature**")
    top_feat = importance.iloc[0]["feature"]
    if top_feat in df_risk.columns:
        fig_box = px.box(
            df_risk, x="risk_tier", y=top_feat,
            color="risk_tier",
            color_discrete_map={"High":"#E24B4A","Medium":"#EF9F27","Low":"#639922"},
            category_orders={"risk_tier":["High","Medium","Low"]},
            labels={"risk_tier":"Risk Tier", top_feat: top_feat},
        )
        fig_box.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info(f"Top feature: **{top_feat}**")


# =============================================================================
# ROW 4: EARLY WARNING LIST  ← the most actionable output
# =============================================================================
st.markdown('<div class="section-header">Early Warning List — Top Customers to Contact Today</div>',
            unsafe_allow_html=True)

st.markdown(
    f"Showing the **top {top_n} highest-risk customers**. "
    "Sort by churn probability, filter by tier, then hand this list to your customer success team."
)

# Filter controls
col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
with col_f1:
    tier_filter = st.multiselect("Filter by tier",
                                  ["High","Medium","Low"],
                                  default=["High","Medium"])
with col_f2:
    min_prob = st.number_input("Min probability (%)", 0, 100, 50)
with col_f3:
    st.write("")  # spacer

# Apply filters
display_cols = ["customer_id", "churn_probability", "risk_tier",
                "monthly_value", "actual_churn"]
if "Customer service calls" in df_risk.columns:
    display_cols.insert(3, "Customer service calls")
if "International plan" in df_risk.columns:
    display_cols.insert(4, "International plan")

available_cols = [c for c in display_cols if c in df_risk.columns]

filtered = df_risk[
    (df_risk["risk_tier"].isin(tier_filter)) &
    (df_risk["churn_probability"] >= min_prob)
].head(top_n)[available_cols].copy()

# Format for display
def color_tier(val):
    colors = {"High":"background-color:#fde8e8;color:#A32D2D;font-weight:600",
              "Medium":"background-color:#fef3e0;color:#854F0B;font-weight:600",
              "Low":"background-color:#eaf3de;color:#3B6D11;font-weight:600"}
    return colors.get(val, "")

def color_prob(val):
    if val >= high_thresh:   return "background-color:#fde8e8;font-weight:600"
    if val >= medium_thresh: return "background-color:#fef3e0"
    return ""

styled = (
    filtered.style
    .applymap(color_tier, subset=["risk_tier"])
    .applymap(color_prob, subset=["churn_probability"])
    .format({"churn_probability": "{:.1f}%", "monthly_value": "${:.2f}"})
)

st.dataframe(styled, use_container_width=True, height=420)

# Download button
csv_dl = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download Early Warning List (CSV)",
    data=csv_dl,
    file_name="early_warning_list.csv",
    mime="text/csv",
)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<small>Model: XGBoost Classifier &nbsp;|&nbsp; "
    "Explainability: SHAP TreeExplainer &nbsp;|&nbsp; "
    "Built as a portfolio project — "
    "<a href='https://github.com'>View on GitHub</a></small>",
    unsafe_allow_html=True
)
