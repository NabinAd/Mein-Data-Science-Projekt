"""Fehlerhafte Eingaben und den echten Prophet-Prognosehorizont prüfen."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting import forecast_store, load_sales, prepare_store_data


def test_loader_filters_closed_days_and_sorts_each_store():
    data = load_sales()
    assert set(data["Store"]) == {1, 2, 3}
    assert data["Open"].eq(1).all()
    assert data["Sales"].gt(0).all()
    assert all(group["Date"].is_monotonic_increasing for _, group in data.groupby("Store"))


@pytest.mark.parametrize("change", ["duplicate", "missing_column", "bad_date", "no_sales"])
def test_loader_rejects_invalid_sales(tmp_path, change):
    data = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": ["2020-01-01", "2020-01-02"],
            "Open": [1, 1],
            "Sales": [10, 20],
        }
    )
    if change == "duplicate":
        data.loc[1, "Date"] = data.loc[0, "Date"]
    elif change == "missing_column":
        data = data.drop(columns="Sales")
    elif change == "bad_date":
        data.loc[0, "Date"] = "not a date"
    else:
        data["Sales"] = 0
    path = tmp_path / "sales.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_sales(path)


@pytest.mark.parametrize("days", [0, -1, 366, 1.5, True])
def test_forecast_rejects_invalid_horizons(days):
    with pytest.raises(ValueError, match="Prognosehorizont"):
        forecast_store(load_sales(), 1, days)


def test_unknown_store_is_reported_before_training():
    with pytest.raises(ValueError, match="mindestens zwei"):
        prepare_store_data(load_sales(), 999)


def test_real_forecast_contains_exactly_requested_future_dates():
    data = load_sales()
    _, result = forecast_store(data, 1, days=7)
    last_date = data.loc[data["Store"] == 1, "Date"].max()
    future = result.loc[result["ds"] > last_date]
    assert (
        future["ds"].tolist() == pd.date_range(last_date + pd.Timedelta(days=1), periods=7).tolist()
    )
    assert future["Store"].eq(1).all()
    assert np.isfinite(future[["yhat", "yhat_lower", "yhat_upper"]]).all().all()
