"""Churn-Logik (aus Chrun_prediction.ipynb)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent   # Projektordner (neben app.py)
CHURN_CSV = ROOT / "data" / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_churn():
    """Telco-Daten laden und bereinigen. customerID bleibt als Spalte erhalten."""
    df = pd.read_csv(CHURN_CSV)
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan).astype(float)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def train_churn_model(df):
    """Logistische Regression trainieren und alles zurückgeben, was app.py braucht."""
    ids = df["customerID"]
    encoded = pd.get_dummies(df.drop(columns=["customerID"]),
                             drop_first=True, dtype=int)
    X = encoded.drop(columns=["Churn"])
    y = encoded["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)

    pred = model.predict(X_test_s)
    proba = pd.Series(model.predict_proba(X_test_s)[:, 1], index=X_test.index)

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    # absteigend sortiert: oben = erhöht Kündigungsrisiko, unten = senkt es
    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)

    return {
        "model": model,
        "metrics": metrics,
        "cm": confusion_matrix(y_test, pred),
        "coefs": coefs,
        "ids": ids.loc[X_test.index],   # customerID der Testkunden
        "proba": proba,
        "y_test": y_test,
    }