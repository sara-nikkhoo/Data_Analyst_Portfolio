"""
Handle new data uploads and score with pre-trained models (Option A: Append without retraining)
"""

import pandas as pd
import numpy as np
import joblib
import os
from loguru import logger

ARTIFACTS_DIR = "artifacts"
DATA_PATH = "online_retail_II.xlsx"
PROFIT_MARGIN = 0.15

def load_pretrained_models():
    """Load all pre-trained models from artifacts."""
    try:
        rfm = joblib.load(f"{ARTIFACTS_DIR}/rfm.pkl")
        meta = joblib.load(f"{ARTIFACTS_DIR}/model_meta.pkl")
        xgb_reg = joblib.load(f"{ARTIFACTS_DIR}/xgb_reg.pkl")
        xgb_clf = joblib.load(f"{ARTIFACTS_DIR}/xgb_clf.pkl")
        bgnbd_params = joblib.load(f"{ARTIFACTS_DIR}/bgnbd_params.pkl")
        ggf_params = joblib.load(f"{ARTIFACTS_DIR}/ggf_params.pkl")
        logger.info("✅ Pre-trained models loaded successfully")
        return rfm, meta, xgb_reg, xgb_clf, bgnbd_params, ggf_params
    except FileNotFoundError as e:
        logger.error(f"❌ Models not found: {e}")
        return None, None, None, None, None, None


def _rebuild_bgnbd(params_dict):
    """Reconstruct BetaGeoFitter from saved parameters."""
    from lifetimes import BetaGeoFitter
    bgf = BetaGeoFitter(penalizer_coef=params_dict["penalizer_coef"])
    bgf.params_ = {k: np.float64(v) for k, v in params_dict["params"].items()}
    bgf._fitted_parameters_names = list(params_dict["params"].keys())
    return bgf


def _rebuild_ggf(params_dict):
    """Reconstruct GammaGammaFitter from saved parameters."""
    from lifetimes import GammaGammaFitter
    ggf = GammaGammaFitter(penalizer_coef=params_dict["penalizer_coef"])
    ggf.params_ = {k: np.float64(v) for k, v in params_dict["params"].items()}
    ggf._fitted_parameters_names = list(params_dict["params"].keys())
    return ggf


def process_new_data(new_df, snapshot):
    """Score new customers with pre-trained models and append to RFM."""
    
    rfm, meta, xgb_reg, xgb_clf, bgnbd_params, ggf_params = load_pretrained_models()
    if rfm is None:
        return None
    
    logger.info(f"Processing {len(new_df):,} new records...")
    
    # Extract features from new data (same as training)
    T0 = snapshot - pd.Timedelta(days=90)
    
    feat_base = (
        new_df.groupby("Customer ID")
        .agg(
            last_purchase=("InvoiceDate", "max"),
            frequency=("Invoice", "nunique"),
            sales_value_sum=("order_value", "sum"),
            sales_value_avg=("order_value", "mean"),
            quantity_avg=("Quantity", "mean"),
        )
        .reset_index()
    )
    feat_base["recency"] = (T0 - feat_base["last_purchase"]).dt.days
    feat_base = feat_base.drop(columns=["last_purchase"])
    
    df_28 = new_df[new_df["InvoiceDate"] >= T0 - pd.Timedelta(days=28)]
    feat_28 = (
        df_28.groupby("Customer ID")
        .agg(txn_last_28d=("Invoice", "nunique"), spend_last_28d=("order_value", "sum"))
        .reset_index()
    )
    
    df_14 = new_df[new_df["InvoiceDate"] >= T0 - pd.Timedelta(days=14)]
    feat_14 = (
        df_14.groupby("Customer ID")
        .agg(txn_last_14d=("Invoice", "nunique"), spend_last_14d=("order_value", "sum"))
        .reset_index()
    )
    
    feat_div = (
        new_df.groupby("Customer ID")
        .agg(n_unique_products=("StockCode", "nunique"),
             n_unique_desc=("Description", "nunique"))
        .reset_index()
    )
    
    features_df = (
        feat_base
        .merge(feat_28, on="Customer ID", how="left")
        .merge(feat_14, on="Customer ID", how="left")
        .merge(feat_div, on="Customer ID", how="left")
        .fillna(0)
    )
    
    FEATURE_COLS = meta["feature_cols"]
    X_new = features_df[FEATURE_COLS].values
    
    # Score with ML models
    pred_spend = np.maximum(xgb_reg.predict(X_new), 0)
    pred_prob = xgb_clf.predict_proba(X_new)[:, 1]
    pred_clv_90d = pred_spend * pred_prob
    
    features_df["pred_spend"] = pred_spend
    features_df["pred_prob"] = pred_prob
    features_df["pred_clv_90d"] = pred_clv_90d
    
    # Score with BG/NBD + Gamma-Gamma
    from lifetimes.utils import summary_data_from_transaction_data
    
    bgf_data = summary_data_from_transaction_data(
        new_df.rename(columns={
            "Customer ID": "customer_id",
            "InvoiceDate": "date",
            "order_value": "monetary_value"
        }),
        customer_id_col="customer_id",
        datetime_col="date",
        monetary_value_col="monetary_value",
        observation_period_end=snapshot
    )
    
    bgf = _rebuild_bgnbd(bgnbd_params)
    bgf_data["predicted_purchases_90d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        90, bgf_data["frequency"], bgf_data["recency"], bgf_data["T"]
    )
    bgf_data["prob_alive"] = bgf.conditional_probability_alive(
        bgf_data["frequency"], bgf_data["recency"], bgf_data["T"]
    )
    
    ggf = _rebuild_ggf(ggf_params)
    gg_data = bgf_data[(bgf_data["frequency"] > 0) & (bgf_data["monetary_value"] > 0)].copy()
    
    if len(gg_data) > 0:
        for months, label in [(3, "90d"), (12, "12m")]:
            clv = ggf.customer_lifetime_value(
                bgf,
                gg_data["frequency"], gg_data["recency"],
                gg_data["T"], gg_data["monetary_value"],
                time=months, discount_rate=0.01
            )
            gg_data[f"clv_{label}"] = clv.values
            gg_data[f"clv_{label}_profit"] = clv.values * PROFIT_MARGIN
    
    logger.info(f"✅ New data scored with {len(features_df):,} customers")
    return features_df, gg_data, bgf_data


def append_to_rfm(new_df, snapshot):
    """Append new scored data to existing RFM table."""
    
    rfm, meta, _, _, _, _ = load_pretrained_models()
    if rfm is None:
        return None
    
    features_df, gg_data, bgf_data = process_new_data(new_df, snapshot)
    if features_df is None:
        return None
    
    # Build RFM for new customers
    new_rfm = (
        new_df.groupby("Customer ID")
        .agg(
            last_purchase=("InvoiceDate", "max"),
            frequency=("Invoice", "nunique"),
            monetary=("order_value", "mean"),
            total_spend=("order_value", "sum"),
            n_orders=("Invoice", "nunique"),
        )
        .reset_index()
    )
    new_rfm["recency"] = (snapshot - new_rfm["last_purchase"]).dt.days
    new_rfm = new_rfm.drop(columns=["last_purchase"])
    
    # Score segments and clusters (use median from existing as baseline)
    new_rfm["r_score"] = pd.qcut(new_rfm["recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
    new_rfm["f_score"] = pd.qcut(new_rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
    new_rfm["m_score"] = pd.qcut(new_rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
    new_rfm["rfm_score"] = new_rfm["r_score"] + new_rfm["f_score"] + new_rfm["m_score"]
    
    # Assign segments
    def map_segment(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r <= 2 and f >= 4 and m >= 4: return "Cannot Lose Them"
        if r >= 4 and f >= 4 and m >= 3: return "Champions"
        if f >= 4 and r >= 3: return "Loyal Customers"
        if r >= 3 and f == 3: return "Potential Loyalists"
        if r >= 4 and f <= 2: return "New Customers"
        if r <= 2 and f >= 3: return "At Risk"
        if r <= 2 and f <= 2: return "Hibernating"
        return "Need Attention"
    
    new_rfm["segment"] = new_rfm.apply(map_segment, axis=1)
    new_rfm["km_cluster"] = -1  # Mark as not clustered
    new_rfm["km_label"] = "New Data"
    
    # Merge ML scores
    new_rfm = new_rfm.merge(features_df[["Customer ID", "pred_spend", "pred_prob", "pred_clv_90d"]], on="Customer ID", how="left")
    
    # Merge BG/NBD scores
    if len(gg_data) > 0:
        gg_cols = ["clv_90d", "clv_12m", "clv_12m_profit", "prob_alive"]
        gg_merge = gg_data[gg_cols].copy()
        gg_merge.index.name = "Customer ID"
        gg_merge = gg_merge.reset_index()
        new_rfm = new_rfm.merge(gg_merge, on="Customer ID", how="left")
    
    bgf_merge = bgf_data[["predicted_purchases_90d"]].copy()
    bgf_merge.index.name = "Customer ID"
    bgf_merge = bgf_merge.reset_index()
    new_rfm = new_rfm.merge(bgf_merge, on="Customer ID", how="left")
    
    # Combine with existing RFM
    combined_rfm = pd.concat([rfm, new_rfm], ignore_index=True)
    
    # Save updated artifacts
    joblib.dump(combined_rfm, f"{ARTIFACTS_DIR}/rfm.pkl")
    logger.success(f"✅ RFM table updated: {len(combined_rfm):,} total customers")
    
    return combined_rfm
