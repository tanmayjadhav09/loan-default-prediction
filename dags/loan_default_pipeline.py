"""
Day 13 : Automating the ML pipeline with Apache Airflow

This DAG runs the complete loan default prediction pipeline:
Task 1 → Load and validate data
Task 2 → Preprocess and clean data
Task 3 → Feature engineering
Task 4 → Train model
Task 5 → Evaluate and save results

Schedule: Every Monday at 6am
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

# ── Default arguments ─────────────────────────────────────────────────────────
# These apply to every task in the DAG
default_args = {
    "owner"           : "tanmay_jadhav",
    "depends_on_past" : False,          # don't wait for previous run
    "start_date"      : datetime(2026, 1, 1),
    "email_on_failure": False,
    "retries"         : 1,              # retry once if task fails
    "retry_delay"     : timedelta(minutes=5),  # wait 5 mins before retry
}

# ── DAG definition ────────────────────────────────────────────────────────────
dag = DAG(
    dag_id="loan_default_prediction",   # unique name for this DAG
    default_args=default_args,
    description="End-to-end loan default prediction pipeline",
    schedule_interval="0 6 * * 1",     # every Monday at 6am
    catchup=False,                      # don't run missed schedules
    tags=["ml", "loan", "prediction"],
)

# ── Task functions ─────────────────────────────────────────────────────────────

def task_load_data(**context):
    """
    Task 1 — Load and validate raw data
    Checks file exists and has expected columns
    """
    import os
    data_path = "C:/Users/Dell/Desktop/loan-default-prediction/data/loan_data.csv"

    log.info("Loading data from: %s", data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Drop unnamed index column
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Validate expected columns
    expected_cols = [
        "SeriousDlqin2yrs",
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "DebtRatio",
        "MonthlyIncome"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    log.info("Data loaded: %d rows x %d columns", df.shape[0], df.shape[1])
    log.info("Class distribution: %s",
             df["SeriousDlqin2yrs"].value_counts().to_dict())

    # Pass data to next task via XCom
    context["ti"].xcom_push(key="row_count", value=df.shape[0])
    return "Data loaded successfully"


def task_preprocess(**context):
    """
    Task 2 — Clean and preprocess data
    Handles nulls, duplicates, outliers
    """
    data_path  = "C:/Users/Dell/Desktop/loan-default-prediction/data/loan_data.csv"
    output_path = "C:/Users/Dell/Desktop/loan-default-prediction/data/loan_data_clean.csv"

    df = pd.read_csv(data_path)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    before = len(df)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Remove invalid rows
    df = df[df["age"] > 0]
    df = df[df["age"] < 110]
    df = df[df["MonthlyIncome"] >= 0]

    # Fill nulls with median
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Cap outliers at 95th percentile
    cols_to_cap = [
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        "MonthlyIncome",
    ]
    for col in cols_to_cap:
        upper = df[col].quantile(0.95)
        df[col] = df[col].clip(upper=upper)

    df.to_csv(output_path, index=False)
    log.info("Preprocessing complete: %d → %d rows", before, len(df))
    return f"Preprocessed: {len(df)} rows saved"


def task_feature_engineering(**context):
    """
    Task 3 — Create new features
    """
    input_path  = "C:/Users/Dell/Desktop/loan-default-prediction/data/loan_data_clean.csv"
    output_path = "C:/Users/Dell/Desktop/loan-default-prediction/data/loan_data_features.csv"

    df = pd.read_csv(input_path)

    # Create new features
    df["MonthlyDebt"] = df["DebtRatio"] * df["MonthlyIncome"]
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["TotalTimesLate"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] +
        df["NumberOfTime60-89DaysPastDueNotWorse"] +
        df["NumberOfTimes90DaysLate"]
    )
    df["HighRiskFlag"] = (
        (df["NumberOfTimes90DaysLate"] >= 1) &
        (df["DebtRatio"] > 0.4)
    ).astype(int)

    # One-hot encode
    df["AgeGroup"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 65, 110],
        labels=["Very Young", "Young", "Middle", "Senior", "Elderly"]
    )
    df["CreditUtilCategory"] = pd.cut(
        df["RevolvingUtilizationOfUnsecuredLines"],
        bins=[-1, 0.3, 0.6, 0.9, float("inf")],
        labels=["Low", "Medium", "High", "Very High"]
    )
    df = pd.get_dummies(df, columns=["AgeGroup", "CreditUtilCategory"],
                        drop_first=True)

    # Fix booleans
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Fix infinity and nulls
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)

    df.to_csv(output_path, index=False)
    log.info("Feature engineering complete: %d columns", df.shape[1])
    return f"Features created: {df.shape[1]} columns"


def task_train_model(**context):
    """
    Task 4 — Train XGBoost model
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    import joblib

    input_path  = "C:/Users/Dell/Desktop/loan-default-prediction/data/loan_data_features.csv"
    model_path  = "C:/Users/Dell/Desktop/loan-default-prediction/models/airflow_xgboost_model.pkl"

    df = pd.read_csv(input_path)

    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    # Save
    joblib.dump(model, model_path)
    log.info("Model trained — ROC-AUC: %.4f", auc)

    # Pass AUC to next task
    context["ti"].xcom_push(key="roc_auc", value=auc)
    return f"Model trained — AUC: {auc:.4f}"


def task_save_results(**context):
    """
    Task 5 — Save pipeline results
    """
    import json
    from datetime import datetime

    # Get AUC from previous task via XCom
    ti    = context["ti"]
    auc   = ti.xcom_pull(task_ids="train_model", key="roc_auc")
    rows  = ti.xcom_pull(task_ids="load_data",   key="row_count")

    results = {
        "run_date"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows_processed": rows,
        "model"      : "XGBoost",
        "roc_auc"    : round(auc, 4),
        "status"     : "success"
    }

    output_path = "C:/Users/Dell/Desktop/loan-default-prediction/outputs/airflow_run_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info("Results saved: %s", results)
    return f"Pipeline complete — AUC: {auc:.4f}"


# ── Define tasks ──────────────────────────────────────────────────────────────

t1_load = PythonOperator(
    task_id="load_data",
    python_callable=task_load_data,
    dag=dag,
)

t2_preprocess = PythonOperator(
    task_id="preprocess_data",
    python_callable=task_preprocess,
    dag=dag,
)

t3_features = PythonOperator(
    task_id="feature_engineering",
    python_callable=task_feature_engineering,
    dag=dag,
)

t4_train = PythonOperator(
    task_id="train_model",
    python_callable=task_train_model,
    dag=dag,
)

t5_results = PythonOperator(
    task_id="save_results",
    python_callable=task_save_results,
    dag=dag,
)

# ── Set task order ────────────────────────────────────────────────────────────
# >> means "then run"
# load → preprocess → features → train → results

t1_load >> t2_preprocess >> t3_features >> t4_train >> t5_results