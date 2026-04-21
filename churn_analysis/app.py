import streamlit as st
import pandas as pd
import joblib
import os

# 1. Page Config & Professional Styling (from dashboard.py)
st.set_page_config(page_title="Churn Early Warning", page_icon="⚡", layout="wide")

st.markdown("""
<style>
  .metric-card { background: #f8f9fa; border-radius: 10px; padding: 1.2rem; border-left: 5px solid #378ADD; margin-bottom: 1rem; }
  .metric-card.danger { border-left-color: #E24B4A; }
  .metric-card.warning { border-left-color: #EF9F27; }
  .metric-card.success { border-left-color: #639922; }
  .metric-label { font-size: 12px; color: #888; font-weight: 500; text-transform: uppercase; }
  .metric-value { font-size: 28px; font-weight: 700; color: #1a1a1a; }
</style>
""", unsafe_allow_html=True)

# 2. Robust Artifact Loading
@st.cache_resource
def load_artifacts():
    artifact = joblib.load("prediction_model.sav")
    return artifact["model"], artifact["features"]


model, features = load_artifacts()

# 3. Sidebar: Configuration
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])
    
    st.markdown("---")
    high_thresh = st.slider("High Risk Threshold (%)", 50, 95, 70)
    med_thresh = st.slider("Medium Risk Threshold (%)", 10, high_thresh, 40)
    
    st.markdown("---")
    top_n = st.number_input("Display Top N Customers", value=50)
    
    # Updated: Filter includes 'Low' by default
    tier_filter = st.multiselect(
        "Filter Tiers to Display", 
        ["High", "Medium", "Low"], 
        default=["High", "Medium", "Low"]
    )

# 4. Main UI Logic
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("pic.svg", width=80)
with col2:
    st.title("Customer Churn Early Warning System")

if model is None:
    st.error("❌ Model file not found! Please run your classifier script first to generate 'prediction_model.sav'.")
else:
    # Determine Data Source: Uploaded vs Default
    df_raw = None
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.info("Using uploaded dataset.")
    elif os.path.exists("churn_test.csv"):
        df_raw = pd.read_csv("churn_test.csv")
        st.caption("Using default dataset: churn.csv")
    
    
    if df_raw is not None:
        # Preprocessing to match feature names
        df = df_raw.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)
        
        # Mapping binary categories
        m = {"yes": 1, "no": 0, "Yes": 1, "No": 0}
        for col in ["international_plan", "voice_mail_plan"]:
            if col in df.columns:
                df[col] = df[col].map(m)
        
        # Prediction
        X = df[features].astype(float)
        probs = model.predict_proba(X)[:, 1]
        
        df["churn_probability_%"] = (probs * 100).round(1)
        # Risk 
        df["risk_tier"] = df["churn_probability_%"].apply(
            lambda x: "High" if x >= high_thresh else ("Medium" if x >= med_thresh else "Low")
        )
        
        # Revenue 
        if "total_day_charge" in df.columns:
            df["monthly_value_usd"] = (df["total_day_charge"] * 3).round(2)
        
        if "customer_id" not in df.columns:
            df.insert(0, "customer_id", [f"CUST-{str(i).zfill(4)}" for i in range(len(df))])

        # 5. KPI Cards (Added Low Risk)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card danger"><div class="metric-label">High Risk</div><div class="metric-value">{len(df[df["risk_tier"]=="High"])}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card warning"><div class="metric-label">Medium Risk</div><div class="metric-value">{len(df[df["risk_tier"]=="Medium"])}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card success"><div class="metric-label">Low Risk</div><div class="metric-value">{len(df[df["risk_tier"]=="Low"])}</div></div>', unsafe_allow_html=True)
        with c4:
            rev_high = df[df["risk_tier"] == "High"]["monthly_value_usd"].sum()
            st.markdown(f'<div class="metric-card"><div class="metric-label">Rev at Risk (High)</div><div class="metric-value">${rev_high:,.0f}</div></div>', unsafe_allow_html=True)

        # 6. Actionable Table
        st.subheader("Early Warning List")
        
        display_cols = ["customer_id", "churn_probability_%", "risk_tier", "monthly_value_usd", 
                        "international_plan", "number_customer_service_calls", "total_day_minutes"]
        
        available_cols = [c for c in display_cols if c in df.columns]
        
        # Apply filters
        filtered_df = df[df["risk_tier"].isin(tier_filter)]
        final_table = filtered_df.sort_values("churn_probability_%", ascending=False).head(top_n)[available_cols]

        # Table Styling
        def color_risk(val):
            if val == "High": return "color: #A32D2D; font-weight: bold;"
            if val == "Medium": return "color: #854F0B;"
            return "color: #3B6D11;"

        st.dataframe(
            final_table.style.map(color_risk, subset=["risk_tier"])
                            .background_gradient(subset=["churn_probability_%"], cmap="YlOrRd"),
            use_container_width=True
        )
    else:
        st.warning("⚠️ No data found. Please upload a CSV file.")
