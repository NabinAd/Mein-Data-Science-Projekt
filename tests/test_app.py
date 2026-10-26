"""Beide Dashboard-Seiten einschließlich Widget-Interaktionen ausführen."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_churn_customer_selection_and_forecast_navigation():
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert not app.error
    assert "Customer Churn" in app.title[0].value
    customer = app.selectbox[0]
    customer.select(customer.options[1]).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Forecast").run()
    assert not app.exception
    assert not app.error
    assert "Umsatzprognose" in app.title[0].value
    app.sidebar.slider[0].set_value(7).run()
    app.sidebar.selectbox[0].set_value(2).run()
    assert not app.exception
    assert not app.error
    assert any("Letzter Datenstand" in caption.value for caption in app.caption)


def test_missing_csv_is_shown_as_a_readable_error(monkeypatch):
    import streamlit as st

    import src.churn

    def missing_file():
        raise FileNotFoundError("Telco-Beispieldatei fehlt")

    st.cache_data.clear()
    st.cache_resource.clear()
    monkeypatch.setattr(src.churn, "load_churn", missing_file)
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert "Telco-Beispieldatei fehlt" in app.error[0].value
    st.cache_data.clear()
    st.cache_resource.clear()
