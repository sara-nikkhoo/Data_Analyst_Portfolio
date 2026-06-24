import streamlit as st
import os
import sys
import pandas as pd

# ── Path resolution
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

st.set_page_config(
    page_title="CLV Intelligence",
    page_icon="💎",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #f4f6f9 !important;
}
[data-testid="stSidebar"] {
    background-color: #1a1f36 !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem !important;
    padding: 0.3rem 0 !important;
}

.card {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    border-left: 5px solid #4f8ef7;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.card.danger  { border-left-color: #e53e3e; }
.card.warning { border-left-color: #dd6b20; }
.card.success { border-left-color: #38a169; }
.card.purple  { border-left-color: #805ad5; }
.card.teal    { border-left-color: #319795; }
.card-label { font-size: 11px; color: #888; font-weight: 600;
               text-transform: uppercase; letter-spacing: 0.06em; }
.card-value { font-size: 26px; font-weight: 700; color: #1a202c; margin-top: 2px; }
.card-sub   { font-size: 12px; color: #a0aec0; margin-top: 2px; }

.section-title {
    font-size: 1.05rem; font-weight: 700; color: #1a202c;
    margin: 1.5rem 0 0.8rem 0;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.4rem;
}
.model-badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
}
.badge-bgnbd { background: #ebf4ff; color: #2b6cb0; }
.badge-ml    { background: #faf5ff; color: #6b46c1; }
.info-box {
    background: #ebf8ff; border-left: 4px solid #4299e1;
    border-radius: 6px; padding: 0.8rem 1rem;
    font-size: 0.82rem; color: #2c5282; margin: 0.5rem 0;
}
.warn-box {
    background: #fffaf0; border-left: 4px solid #ed8936;
    border-radius: 6px; padding: 0.8rem 1rem;
    font-size: 0.82rem; color: #7b341e; margin: 0.5rem 0;
}
[data-testid="stDataFrame"] { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.2rem 0 2rem 0;'>
        <div style='font-size:1.2rem; font-weight:800; color:#90cdf4; letter-spacing:0.05em;'>
        CUSTOMER VALUE PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:0.8rem; font-weight:600; color:#a0aec0;'>📤 DATA UPLOAD</div>",
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "Upload new transaction data",
        type=["csv", "xlsx"],
        key="data_upload",
        help="Upload CSV or Excel file with columns: Customer ID, InvoiceDate, Quantity, Price, etc."
    )
    if uploaded_file:
        try:
            # Read uploaded file
            if uploaded_file.name.endswith('.csv'):
                new_data = pd.read_csv(uploaded_file)
            else:
                new_data = pd.read_excel(uploaded_file)
            
            # Required columns check
            required_cols = {"Customer ID", "InvoiceDate", "Invoice", "Quantity", "Price"}
            if not required_cols.issubset(new_data.columns):
                st.error(f"❌ Missing columns. Required: {required_cols}")
            else:
                # Parse dates and create order_value
                new_data["InvoiceDate"] = pd.to_datetime(new_data["InvoiceDate"])
                new_data["order_value"] = new_data["Quantity"] * new_data["Price"]
                
                st.success(f"✅ File loaded: {uploaded_file.name} ({len(new_data):,} rows)")
                
                if st.button("🔄 Append & Score New Data", key="append_btn"):
                    from data_handler import append_to_rfm
                    snapshot = new_data["InvoiceDate"].max()
                    with st.spinner("Scoring new customers with pre-trained models..."):
                        updated_rfm = append_to_rfm(new_data, snapshot)
                        if updated_rfm is not None:
                            st.success(f"✅ Data appended! Total customers now: {len(updated_rfm):,}")
                            st.info("Refresh the page to see updated analytics")
                        else:
                            st.error("❌ Error processing data")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["  RFM Segmentation", "  CLV Prediction"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    
    st.markdown("---")

# ── Route
if "RFM" in page:
    from views.rfm import show
    show()
else:
    from views.clv import show
    show()