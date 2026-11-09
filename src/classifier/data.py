"""Loan default data: the features, preprocessing and split from the 31250 pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "data" / "Assignment3-Loan-Dataset.csv"
TARGET = "loan_default"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "age", "income", "credit_score", "loan_amount", "debt_to_income_pct",
    "num_dependents", "years_employed", "account_balance", "num_prev_loans",
    "monthly_expenses", "application_year", "loan_duration_months",
    "loan_to_income", "monthly_payment_est", "payment_to_income",
    "disposable_income", "expense_to_income", "balance_to_loan",
    "credit_dti_interaction", "credit_loan_interaction", "low_credit_flag",
    "high_dti_flag", "long_duration_flag", "many_prev_loans_flag",
    "high_loan_flag", "risk_flag_count",
]
CATEGORICAL_FEATURES = ["gender", "employment_status", "loan_purpose", "region", "marital_status"]
ORDINAL_FEATURES = ["education_level"]
EDUCATION_ORDER = [["High School", "Bachelor", "Master", "PhD"]]


def engineer_features(df):
    df = df.copy()
    df["loan_to_income"] = df["loan_amount"] / (df["income"] + 1)
    df["monthly_payment_est"] = df["loan_amount"] / df["loan_duration_months"]
    df["payment_to_income"] = df["monthly_payment_est"] / ((df["income"] / 12) + 1)
    df["disposable_income"] = (df["income"] / 12) - df["monthly_expenses"]
    df["expense_to_income"] = df["monthly_expenses"] / ((df["income"] / 12) + 1)
    df["balance_to_loan"] = df["account_balance"] / (df["loan_amount"] + 1)
    df["credit_dti_interaction"] = df["credit_score"] * df["debt_to_income_pct"]
    df["credit_loan_interaction"] = df["credit_score"] / (df["loan_amount"] + 1) * 1000

    df["low_credit_flag"] = (df["credit_score"] < 582).astype(int)
    df["high_dti_flag"] = (df["debt_to_income_pct"] > 36).astype(int)
    df["long_duration_flag"] = (df["loan_duration_months"] >= 60).astype(int)
    df["many_prev_loans_flag"] = (df["num_prev_loans"] >= 6).astype(int)
    df["high_loan_flag"] = (df["loan_amount"] > 26000).astype(int)
    df["risk_flag_count"] = (
        df["low_credit_flag"] + df["high_dti_flag"] + df["long_duration_flag"]
        + df["many_prev_loans_flag"] + df["high_loan_flag"]
    )
    return df.replace([np.inf, -np.inf], np.nan)


def load_split(csv_path=CSV_PATH):
    """Stratified 80/20 split, returned as raw (unfitted) feature frames."""
    df = engineer_features(pd.read_csv(csv_path))
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)


def build_preprocessor(scale=True):
    """Unfitted preprocessor; fit it on the training split only."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="High School")),
        ("ordinal", OrdinalEncoder(categories=EDUCATION_ORDER,
                                   handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
        ("ord", ordinal, ORDINAL_FEATURES),
    ])
