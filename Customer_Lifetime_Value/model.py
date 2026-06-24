"""
train_pipeline.py — Run once before launching the dashboard.

Usage:
    python train_pipeline.py

Output:
    artifacts/rfm.pkl           → RFM table with segments
    artifacts/bgnbd_params.pkl  → BG/NBD model parameters (dict)
    artifacts/ggf_params.pkl    → Gamma-Gamma model parameters (dict)
    artifacts/xgb_reg.pkl       → XGBoost Tweedie regression model
    artifacts/xgb_clf.pkl       → XGBoost purchase classifier
    artifacts/model_meta.pkl    → metadata (features, threshold, scores, stats)
    artifacts/top_products.pkl  → top products for high-CLV customers
"""

import pandas as pd
import numpy as np
import joblib
import os
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, roc_auc_score,
    f1_score, roc_curve
)
from xgboost import XGBRegressor, XGBClassifier
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

ARTIFACTS_DIR      = "artifacts"
DATA_PATH          = "online_retail_II.xlsx"
PROFIT_MARGIN      = 0.15
PREDICTION_HORIZON = 90   # days
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    logger.info("Loading data...")
    df09 = pd.read_excel(path, sheet_name="Year 2009-2010")
    df10 = pd.read_excel(path, sheet_name="Year 2010-2011")
    df   = pd.concat([df09, df10], ignore_index=True)
    logger.info(f"Raw shape: {df.shape}")

    # Drop missing Customer ID
    df = df.dropna(subset=["Customer ID"])

    # Parse types
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = df["Customer ID"].astype(int).astype(str)

    # Remove returns
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # Remove non-positive values
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()

    # Order value
    df["order_value"] = df["Quantity"] * df["Price"]

    logger.info(f"Clean shape: {df.shape} | Customers: {df['Customer ID'].nunique():,}")
    return df


# ─────────────────────────────────────────────
# 2. WHOLESALE DETECTION
# ─────────────────────────────────────────────
def detect_wholesale(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Detecting wholesale customers...")

    summary = (
        df.groupby("Customer ID")
        .agg(avg_order_value=("order_value", "mean"),
             avg_quantity   =("Quantity",    "mean"))
        .reset_index()
    )

    def iqr_upper(series, k=3):
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        return Q3 + k * (Q3 - Q1)

    val_thresh = iqr_upper(summary["avg_order_value"])
    qty_thresh = iqr_upper(summary["avg_quantity"])

    wholesale_ids = set(summary[
        (summary["avg_order_value"] > val_thresh) |
        (summary["avg_quantity"]    > qty_thresh)
    ]["Customer ID"])

    df_retail = df[~df["Customer ID"].isin(wholesale_ids)].copy()
    logger.info(f"Wholesale removed: {len(wholesale_ids):,} | Retail customers: {df_retail['Customer ID'].nunique():,}")
    return df_retail


# ─────────────────────────────────────────────
# 3. RFM + SEGMENTS
# ─────────────────────────────────────────────
def build_rfm(df: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    logger.info("Building RFM table...")

    rfm = (
        df.groupby("Customer ID")
        .agg(
            last_purchase=("InvoiceDate", "max"),
            frequency    =("Invoice",     "nunique"),
            monetary     =("order_value", "mean"),
            total_spend  =("order_value", "sum"),
            n_orders     =("Invoice",     "nunique"),
        )
        .reset_index()
    )
    rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days
    rfm = rfm.drop(columns=["last_purchase"])

    # Score 1-5
    rfm["r_score"] = pd.qcut(rfm["recency"], q=5, labels=[5,4,3,2,1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"),  q=5, labels=[1,2,3,4,5]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # Segments
    def map_segment(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r <= 2 and f >= 4 and m >= 4: return "Cannot Lose Them"
        if r >= 4 and f >= 4 and m >= 3: return "Champions"
        if f >= 4 and r >= 3:            return "Loyal Customers"
        if r >= 3 and f == 3:            return "Potential Loyalists"
        if r >= 4 and f <= 2:            return "New Customers"
        if r <= 2 and f >= 3:            return "At Risk"
        if r <= 2 and f <= 2:            return "Hibernating"
        return "Need Attention"

    rfm["segment"] = rfm.apply(map_segment, axis=1)
    logger.info(f"RFM built: {len(rfm):,} customers | {rfm['segment'].nunique()} segments")
    return rfm


# ─────────────────────────────────────────────
# 4. K-MEANS CLUSTERING
# ─────────────────────────────────────────────
def add_clusters(rfm: pd.DataFrame) -> pd.DataFrame:
    logger.info("Fitting K-Means clusters...")
    from sklearn.metrics import silhouette_score

    X = rfm[["recency", "frequency", "monetary"]].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # Select K: compare 4 and 5
    best_k, best_sil = 4, -1
    for k in [4, 5]:
        km  = KMeans(n_clusters=k, n_init=20, random_state=42)
        lbl = km.fit_predict(X_sc)
        sil = silhouette_score(X_sc, lbl, sample_size=min(5000, len(X_sc)), random_state=42)
        if sil > best_sil:
            best_sil, best_k = sil, k

    km_final = KMeans(n_clusters=best_k, n_init=20, max_iter=500, random_state=42)
    rfm["km_cluster"] = km_final.fit_predict(X_sc)

    profile   = rfm.groupby("km_cluster")["rfm_score"].mean().sort_values(ascending=False)
    tier_names = {profile.index[i]: t for i, t in
                  enumerate(["High Value","Mid-High","Mid-Low","Low Value"][:best_k])}
    rfm["km_label"] = rfm["km_cluster"].map(tier_names)

    logger.info(f"K-Means: K={best_k}, Silhouette={best_sil:.4f}")
    return rfm


# ─────────────────────────────────────────────
# 5. BG/NBD + GAMMA-GAMMA
# ─────────────────────────────────────────────
def fit_probabilistic(df: pd.DataFrame, snapshot: pd.Timestamp) -> dict:
    logger.info("Fitting BG/NBD + Gamma-Gamma...")

    bgf_data = summary_data_from_transaction_data(
        df.rename(columns={
            "Customer ID": "customer_id",
            "InvoiceDate": "date",
            "order_value": "monetary_value"
        }),
        customer_id_col   ="customer_id",
        datetime_col      ="date",
        monetary_value_col="monetary_value",
        observation_period_end=snapshot
    )

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(bgf_data["frequency"], bgf_data["recency"], bgf_data["T"])

    bgf_data["predicted_purchases_90d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        90, bgf_data["frequency"], bgf_data["recency"], bgf_data["T"]
    )
    bgf_data["prob_alive"] = bgf.conditional_probability_alive(
        bgf_data["frequency"], bgf_data["recency"], bgf_data["T"]
    )

    # Gamma-Gamma
    gg_data = bgf_data[(bgf_data["frequency"] > 0) & (bgf_data["monetary_value"] > 0)].copy()
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(gg_data["frequency"], gg_data["monetary_value"])

    for months, label in [(3, "90d"), (12, "12m")]:
        clv = ggf.customer_lifetime_value(
            bgf,
            gg_data["frequency"], gg_data["recency"],
            gg_data["T"],         gg_data["monetary_value"],
            time=months, discount_rate=0.01
        )
        gg_data[f"clv_{label}"]        = clv.values
        gg_data[f"clv_{label}_profit"] = clv.values * PROFIT_MARGIN

    logger.info(f"BG/NBD done | Gamma-Gamma on {len(gg_data):,} repeat buyers")
    return {"bgf": bgf, "ggf": ggf, "bgf_data": bgf_data, "gg_data": gg_data}


# ─────────────────────────────────────────────
# 6. ML CLV (TIME-BASED SPLIT)
# ─────────────────────────────────────────────
def fit_ml_clv(df: pd.DataFrame, snapshot: pd.Timestamp) -> dict:
    logger.info("Fitting ML CLV models (time-based split)...")

    T0     = snapshot - pd.Timedelta(days=PREDICTION_HORIZON)
    df_in  = df[df["InvoiceDate"] <= T0].copy()
    df_out = df[df["InvoiceDate"] >  T0].copy()

    # Targets
    targets = (
        df_out.groupby("Customer ID")
        .agg(sales_90_value=("order_value", "sum"))
        .reset_index()
    )
    targets["purchased_flag"] = 1

    # Features
    feat_base = (
        df_in.groupby("Customer ID")
        .agg(
            last_purchase   =("InvoiceDate",  "max"),
            frequency       =("Invoice",      "nunique"),
            sales_value_sum =("order_value",  "sum"),
            sales_value_avg =("order_value",  "mean"),
            quantity_avg    =("Quantity",     "mean"),
        )
        .reset_index()
    )
    feat_base["recency"] = (T0 - feat_base["last_purchase"]).dt.days
    feat_base = feat_base.drop(columns=["last_purchase"])

    df_28 = df_in[df_in["InvoiceDate"] >= T0 - pd.Timedelta(days=28)]
    feat_28 = (
        df_28.groupby("Customer ID")
        .agg(txn_last_28d=("Invoice","nunique"), spend_last_28d=("order_value","sum"))
        .reset_index()
    )

    df_14 = df_in[df_in["InvoiceDate"] >= T0 - pd.Timedelta(days=14)]
    feat_14 = (
        df_14.groupby("Customer ID")
        .agg(txn_last_14d=("Invoice","nunique"), spend_last_14d=("order_value","sum"))
        .reset_index()
    )

    feat_div = (
        df_in.groupby("Customer ID")
        .agg(n_unique_products=("StockCode","nunique"),
             n_unique_desc    =("Description","nunique"))
        .reset_index()
    )

    features_df = (
        feat_base
        .merge(feat_28, on="Customer ID", how="left")
        .merge(feat_14, on="Customer ID", how="left")
        .merge(feat_div,on="Customer ID", how="left")
        .fillna(0)
    )

    FEATURE_COLS = [c for c in features_df.columns if c != "Customer ID"]

    model_df = features_df.merge(targets, on="Customer ID", how="left")
    model_df["sales_90_value"] = model_df["sales_90_value"].fillna(0)
    model_df["purchased_flag"] = model_df["purchased_flag"].fillna(0).astype(int)

    X   = model_df[FEATURE_COLS].values
    y_r = model_df["sales_90_value"].values
    y_c = model_df["purchased_flag"].values

    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_r, y_c, test_size=0.2, random_state=42
    )

    # ── Regression (Tweedie)
    xgb_reg = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:tweedie", tweedie_variance_power=1.5,
        random_state=42, n_jobs=-1
    )
    cv_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(xgb_reg, X_train, yr_train, cv=cv_kf, scoring="r2")
    xgb_reg.fit(X_train, yr_train)
    yr_pred   = np.maximum(xgb_reg.predict(X_test), 0)
    r2_test   = r2_score(yr_test, yr_pred)
    mae_test  = mean_absolute_error(yr_test, yr_pred)
    logger.info(f"Regression  CV R²={cv_r2.mean():.4f} | Test R²={r2_test:.4f} | MAE=£{mae_test:.2f}")

    # ── Classification
    neg, pos = (yc_train==0).sum(), (yc_train==1).sum()
    xgb_clf = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg/pos, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc   = cross_val_score(xgb_clf, X_train, yc_train, cv=cv_strat, scoring="roc_auc")
    xgb_clf.fit(X_train, yc_train)
    yc_prob = xgb_clf.predict_proba(X_test)[:,1]

    # Optimal threshold
    fpr, tpr, thresholds = roc_curve(yc_test, yc_prob)
    opt_idx   = (tpr - fpr).argmax()
    opt_thresh = round(float(thresholds[opt_idx]), 3)
    auc_test   = roc_auc_score(yc_test, yc_prob)
    f1_test    = f1_score(yc_test, (yc_prob >= opt_thresh).astype(int))
    logger.info(f"Classifier  CV AUC={cv_auc.mean():.4f} | Test AUC={auc_test:.4f} | F1={f1_test:.4f} | Threshold={opt_thresh}")

    # ── Score all customers
    X_all = features_df[FEATURE_COLS].values
    features_df["pred_spend"]   = np.maximum(xgb_reg.predict(X_all), 0)
    features_df["pred_prob"]    = xgb_clf.predict_proba(X_all)[:,1]
    features_df["pred_clv_90d"] = features_df["pred_spend"] * features_df["pred_prob"]
    scores_df = features_df[["Customer ID","pred_spend","pred_prob","pred_clv_90d"]]

    # ── Top products for high-CLV customers
    top_ids = set(features_df.nlargest(100, "pred_clv_90d")["Customer ID"])
    top_products = (
        df_in[df_in["Customer ID"].isin(top_ids)]
        .groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    return {
        "xgb_reg":      xgb_reg,
        "xgb_clf":      xgb_clf,
        "feature_cols": FEATURE_COLS,
        "threshold":    opt_thresh,
        "scores":       scores_df,
        "top_products": top_products,
        "metrics": {
            "reg_cv_r2":  round(cv_r2.mean(), 4),
            "reg_test_r2": round(r2_test, 4),
            "reg_mae":    round(mae_test, 2),
            "clf_cv_auc": round(cv_auc.mean(), 4),
            "clf_auc":    round(auc_test, 4),
            "clf_f1":     round(f1_test, 4),
            "threshold":  opt_thresh,
        }
    }


# ─────────────────────────────────────────────
# 7. SAVE ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(rfm, prob_bundle, ml_bundle, snapshot, df_retail):
    logger.info("Saving artifacts...")

    # Merge all CLV scores onto rfm
    gg = prob_bundle["gg_data"][["clv_90d","clv_12m","clv_12m_profit","prob_alive"]].copy()
    gg.index.name = "Customer ID"
    gg = gg.reset_index()
    rfm = rfm.merge(gg, on="Customer ID", how="left")
    rfm = rfm.merge(prob_bundle["bgf_data"][["predicted_purchases_90d"]]
                    .rename_axis("Customer ID").reset_index(),
                    on="Customer ID", how="left")
    rfm = rfm.merge(ml_bundle["scores"], on="Customer ID", how="left")

    # Descriptive stats
    total_days  = (df_retail["InvoiceDate"].max() - df_retail["InvoiceDate"].min()).days
    total_years = total_days / 365
    cust_agg    = (
        df_retail.groupby("Customer ID")
        .agg(first_purchase=("InvoiceDate","min"),
             last_purchase =("InvoiceDate","max"),
             n_orders      =("Invoice",    "nunique"))
        .reset_index()
    )
    cust_agg["days_since_last"]  = (snapshot - cust_agg["last_purchase"]).dt.days
    cust_agg["observation_days"] = (cust_agg["last_purchase"] - cust_agg["first_purchase"]).dt.days
    cust_agg["avg_interpurchase"] = np.where(
        cust_agg["n_orders"] > 1,
        cust_agg["observation_days"] / (cust_agg["n_orders"] - 1),
        total_days
    )
    cust_agg["is_churned"] = cust_agg["days_since_last"] > (cust_agg["avg_interpurchase"] * 2)
    churn_rate = cust_agg["is_churned"].mean()

    # ── BG/NBD and Gamma-Gamma: save parameters only (not model objects)
    # joblib cannot serialize lifetimes models — they contain unpicklable
    # lambda functions created inside fit(). Save the four parameter values
    # as a plain dict instead. Dashboard rebuilds models from these at load time.
    bgf = prob_bundle["bgf"]
    ggf = prob_bundle["ggf"]

    bgnbd_params = {
        "penalizer_coef": 0.01,
        "params": {k: float(v) for k, v in bgf.params_.items()},
    }
    ggf_params = {
        "penalizer_coef": 0.01,
        "params": {k: float(v) for k, v in ggf.params_.items()},
    }

    # Save all artifacts
    joblib.dump(rfm,                        f"{ARTIFACTS_DIR}/rfm.pkl")
    joblib.dump(bgnbd_params,               f"{ARTIFACTS_DIR}/bgnbd_params.pkl")
    joblib.dump(ggf_params,                 f"{ARTIFACTS_DIR}/ggf_params.pkl")
    joblib.dump(ml_bundle["xgb_reg"],       f"{ARTIFACTS_DIR}/xgb_reg.pkl")
    joblib.dump(ml_bundle["xgb_clf"],       f"{ARTIFACTS_DIR}/xgb_clf.pkl")
    joblib.dump(ml_bundle["top_products"],  f"{ARTIFACTS_DIR}/top_products.pkl")

    meta = {
        "snapshot":       str(snapshot.date()),
        "n_customers":    len(rfm),
        "n_segments":     rfm["segment"].nunique(),
        "total_revenue":  float(rfm["total_spend"].sum()),
        "churn_rate":     round(churn_rate, 4),
        "feature_cols":   ml_bundle["feature_cols"],
        "threshold":      ml_bundle["threshold"],
        "ml_metrics":     ml_bundle["metrics"],
        "profit_margin":  PROFIT_MARGIN,
    }
    joblib.dump(meta, f"{ARTIFACTS_DIR}/model_meta.pkl")

    logger.success("All artifacts saved.")
    logger.info(f"  rfm.pkl            → {len(rfm):,} customers")
    logger.info(f"  bgnbd_params.pkl   → {bgnbd_params['params']}")
    logger.info(f"  ggf_params.pkl     → {ggf_params['params']}")
    logger.info(f"  Reg R²={meta['ml_metrics']['reg_test_r2']}  "
                f"Clf AUC={meta['ml_metrics']['clf_auc']}")
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("CLV TRAINING PIPELINE — Online Retail II")
    logger.info("=" * 55)

    df       = load_and_clean(DATA_PATH)
    df_retail = detect_wholesale(df)
    snapshot  = df_retail["InvoiceDate"].max()

    rfm          = build_rfm(df_retail, snapshot)
    rfm          = add_clusters(rfm)
    prob_bundle  = fit_probabilistic(df_retail, snapshot)
    ml_bundle    = fit_ml_clv(df_retail, snapshot)

    save_artifacts(rfm, prob_bundle, ml_bundle, snapshot, df_retail)

    logger.success("Pipeline complete. Run: streamlit run app.py")