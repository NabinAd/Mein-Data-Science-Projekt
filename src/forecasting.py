"""Historische Rossmann-Umsätze prüfen und pro Filiale mit Prophet modellieren."""

from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

ROOT = Path(__file__).resolve().parent.parent
SALES_CSV = ROOT / "data" / "rossmann" / "train_filtered.csv"


def load_sales(path: str | Path = SALES_CSV) -> pd.DataFrame:
    """Nur geöffnete Tage mit positivem Umsatz für das Modell verwenden."""
    df = pd.read_csv(path, low_memory=False)
    required = {"Store", "Date", "Open", "Sales"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Rossmann-Datei: fehlende Spalten: {', '.join(sorted(missing))}")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="raise")
    for column in ["Store", "Open", "Sales"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if df[list(required)].isna().any().any():
        raise ValueError("Rossmann-Datei enthält fehlende Pflichtwerte.")
    if not np.isfinite(df[["Store", "Open", "Sales"]]).all().all():
        raise ValueError("Rossmann-Datei enthält nicht endliche Zahlen.")
    if not df["Open"].isin([0, 1]).all():
        raise ValueError("Open darf nur 0 oder 1 enthalten.")
    if df.duplicated(["Store", "Date"]).any():
        raise ValueError("Rossmann-Datei enthält mehrere Zeilen pro Filiale und Tag.")
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
    if df.empty:
        raise ValueError("Keine geöffneten Verkaufstage mit positivem Umsatz vorhanden.")
    return df.sort_values(["Store", "Date"])


def prepare_store_data(df: pd.DataFrame, store_id: int) -> pd.DataFrame:
    """Eine Filiale in Prophets Spalten ds (Datum) und y (Umsatz) übersetzen."""
    data = (
        df.loc[df["Store"] == store_id, ["Date", "Sales"]]
        .rename(columns={"Date": "ds", "Sales": "y"})
        .sort_values("ds")
    )
    if len(data) < 2 or data["ds"].nunique() < 2:
        raise ValueError(f"Filiale {store_id} benötigt mindestens zwei verschiedene Verkaufstage.")
    return data


def forecast_store(df: pd.DataFrame, store_id: int, days: int = 30) -> tuple:
    """Auf Historie trainieren und 1 bis 365 weitere Kalendertage prognostizieren."""
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 365:
        raise ValueError("Der Prognosehorizont muss eine ganze Zahl zwischen 1 und 365 sein.")
    data = prepare_store_data(df, store_id)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(data, seed=42)
    forecast = model.predict(model.make_future_dataframe(periods=days))
    forecast["Store"] = store_id
    return model, forecast
