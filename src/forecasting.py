"""Forecasting-Logik (aus forecasting_mit_Prophet.ipynb)."""

from pathlib import Path

import pandas as pd
from prophet import Prophet

ROOT = Path(__file__).resolve().parent.parent
SALES_CSV = ROOT / "data" / "rossmann" / "train_filtered.csv"


def load_sales():
    """Rossmann-Daten laden; nur Tage, an denen geöffnet war."""
    df = pd.read_csv(SALES_CSV, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[(df["Open"] == 1) & (df["Sales"] > 0)]


def forecast_store(df, store_id, days=30):
    """Prophet für einen Store trainieren und `days` Tage vorhersagen."""
    data = (
        df[df["Store"] == store_id][["Date", "Sales"]]
        .rename(columns={"Date": "ds", "Sales": "y"})
        .sort_values("ds")
    )
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False)
    model.fit(data)
    fc = model.predict(model.make_future_dataframe(periods=days))
    fc["Store"] = store_id
    return model, fc
