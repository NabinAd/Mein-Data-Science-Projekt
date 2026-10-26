"""Telco-Daten laden und eine Churn-Pipeline auf getrennten Daten auswerten."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CHURN_CSV = ROOT / "data" / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_churn(path: str | Path = CHURN_CSV) -> pd.DataFrame:
    """CSV prüfen und Typen konvertieren; Median erst beim Training lernen."""
    df = pd.read_csv(path)
    required = {"customerID", "Churn", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Telco-Datei: fehlende Spalten: {', '.join(sorted(missing))}")
    if df.empty or df["customerID"].isna().any() or df["customerID"].duplicated().any():
        raise ValueError("Telco-Datei benötigt eindeutige Kunden-IDs und mindestens eine Zeile.")
    if not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Churn darf nur 'Yes' oder 'No' enthalten.")
    for column in NUMERIC_FEATURES:
        values = df[column].replace(r"^\s*$", np.nan, regex=True)
        df[column] = pd.to_numeric(values, errors="raise")
        if np.isinf(df[column]).any() or df[column].notna().sum() == 0:
            raise ValueError(f"Telco-Datei: ungültige Zahlen in {column}.")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def build_churn_pipeline() -> Pipeline:
    """Vorverarbeitung und Modell teilen dieselbe fit/predict-Schnittstelle."""
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                drop="first",
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocess", preprocessing),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def classification_metrics(y_true, predictions, probabilities) -> dict[str, float]:
    """Kennzahlen für Dashboard und Notebook einheitlich berechnen."""
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1-Score": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
    }


def train_churn_model(df: pd.DataFrame) -> dict:
    """80/20-Split vor allen gelernten Transformationen; IDs sind keine Features."""
    features = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = df["Churn"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    model = build_churn_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = pd.Series(model.predict_proba(x_test)[:, 1], index=x_test.index)
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = pd.Series(
        model.named_steps["classifier"].coef_[0],
        index=feature_names,
    ).sort_values(ascending=False)
    return {
        "model": model,
        "metrics": classification_metrics(y_test, predictions, probabilities),
        "baseline_accuracy": accuracy_score(y_test, baseline.predict(x_test)),
        "cm": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "coefs": coefficients,
        "ids": df.loc[x_test.index, "customerID"],
        "proba": probabilities,
        "y_test": y_test,
    }
