import pandas as pd
import joblib
from loguru import logger
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def train_pipeline():
    logger.info("Modules loaded")

    # 1. Load Data
    try:
        # Using the exact file name from your analysis script
        df = pd.read_csv("churn.csv")
        logger.info("Data loaded from churn.csv")
    except FileNotFoundError:
        logger.warning("churn.csv not found")
        
    # 2. Preprocessing 
    # Manual mapping for Yes/No columns as they appear in the dataset
    yes_no_map = {"yes": 1, "no": 0, "Yes": 1, "No": 0}
    df["international_plan"] = df["international_plan"].map(yes_no_map)
    df["voice_mail_plan"] = df["voice_mail_plan"].map(yes_no_map)
    
    # Ensure target is integer (bool -> int)
    df["churn"] = df["churn"].map({"yes": 1, "no": 0}).astype(int)
    
    # Drop non-predictive columns identified in your research
    df.drop(columns=["state", "area_code", "phone_number"], inplace=True, errors="ignore")
    logger.info("Corrected columns processed and identifiers dropped")

    # 3. Split & Train (80/20 split)
    X = df.drop(columns=["churn"])
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. XGBoost Config 
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = neg_count / pos_count
    
    model = XGBClassifier(
        n_estimators=400, 
        max_depth=5, 
        learning_rate=0.05,
        subsample=0.8,       
        colsample_bytree=0.8,  
        eval_metric="logloss", 
        use_label_encoder=False, 
        random_state=42
    )
    
    model.fit(X_train, y_train)
    logger.info("XGBoost trained with exact 'bigml' feature names")

    # 5. Save Artifacts
    # Saving features ensures the Streamlit app uses the exact same column order
    joblib.dump({"model": model, "features": list(X.columns)}, "prediction_model.sav")
    
    # Save X_test as CSV for testing purposes
    X_test.to_csv("X_test.csv", index=False)
    logger.info("X_test saved to X_test.csv")
    
    logger.info("ML model and feature list saved to prediction_model.sav")

if __name__ == "__main__":
    train_pipeline()
