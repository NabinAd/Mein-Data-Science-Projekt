"""Streamlit-Dashboard: Customer Churn + Umsatzprognose.

Start:  streamlit run app.py
Streamlit führt dieses Skript bei jeder Interaktion von oben nach unten neu
aus. Teure Berechnungen merken wir uns deshalb mit Caching.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.churn import load_churn, train_churn_model
from src.forecasting import forecast_store, load_sales

# Muss der erste Streamlit-Befehl sein
st.set_page_config(page_title="Churn & Forecast", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Farben (an einer Stelle definiert, überall wiederverwendet)
# ---------------------------------------------------------------------------
BLUE, RED, GREEN, ORANGE, GRAY = "#1F6FEB", "#E5534B", "#2DA44E", "#F0883E", "#8B95A5"
STATUS_COLORS = {"Geblieben": GREEN, "Gekündigt": RED}

# KPI-Karten und Feinschliff per CSS
st.markdown(
    """
    <style>
    .kpi {background:#fff; border-radius:12px; padding:16px 18px;
          border-left:6px solid var(--c); box-shadow:0 1px 6px rgba(0,0,0,.08);}
    .kpi .label {font-size:0.85rem; color:#6B7685; margin-bottom:2px;}
    .kpi .value {font-size:1.8rem; font-weight:700; color:#1B2430; line-height:1.2;}
    .kpi .sub   {font-size:0.8rem; color:#6B7685; margin-top:2px;}
    .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(col, label, value, sub="", color=BLUE):
    """Zeichnet eine KPI-Karte in die übergebene Spalte."""
    col.markdown(
        f'<div class="kpi" style="--c:{color}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}&nbsp;</div></div>',
        unsafe_allow_html=True,
    )


def style(fig, height=380):
    """Einheitliches Aussehen für alle Plotly-Diagramme."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E3E8EF")
    return fig


def to_csv(df):
    """CSV für deutsches Excel: Semikolon als Trenner, Komma als Dezimalzeichen."""
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


# ---------------------------------------------------------------------------
# Caching: Daten laden / Modelle trainieren nur einmal, nicht bei jedem Klick
# ---------------------------------------------------------------------------
@st.cache_data
def get_churn_data():
    return load_churn()


@st.cache_resource
def get_churn_result():
    return train_churn_model(get_churn_data())


@st.cache_data
def get_sales():
    return load_sales()


@st.cache_data
def get_forecast(store_id, days):
    _, forecast = forecast_store(get_sales(), store_id, days)
    return forecast


# ---------------------------------------------------------------------------
# Seite 1: Customer Churn
# ---------------------------------------------------------------------------
def page_churn():
    st.title("📉 Customer Churn")
    st.markdown(
        "Welche Telekommunikationskunden kündigen bald? Ein Logistic-Regression-"
        "Modell schätzt für jeden Kunden die **Kündigungswahrscheinlichkeit**."
    )
    df = get_churn_data()
    res = get_churn_result()
    m = res["metrics"]

    # --- KPI-Karten ---
    c = st.columns(5)
    kpi(c[0], "Kunden", f"{len(df):,}".replace(",", "."), "im Datensatz", BLUE)
    kpi(c[1], "Churn-Rate", f"{df['Churn'].mean():.1%}", "haben gekündigt", RED)
    kpi(c[2], "Accuracy", f"{m['Accuracy']:.1%}", "Anteil richtiger Vorhersagen", GREEN)
    kpi(c[3], "Recall", f"{m['Recall']:.1%}", "erkannte Kündiger", ORANGE)
    kpi(c[4], "ROC-AUC", f"{m['ROC-AUC']:.2f}", "Trennschärfe (1 = perfekt)", BLUE)
    st.write("")

    tab1, tab2, tab3 = st.tabs(["🔍 Übersicht", "👤 Kundenprognose", "🧪 Modell"])

    # --- Tab 1: Übersicht ---
    with tab1:
        st.caption("Wie verteilen sich die Kündigungen im Datensatz?")
        d = df.assign(Status=df["Churn"].map({1: "Gekündigt", 0: "Geblieben"}))
        left, right = st.columns(2)

        fig = px.pie(d, names="Status", hole=0.55, color="Status",
                     color_discrete_map=STATUS_COLORS, title="Gekündigt vs. geblieben")
        left.plotly_chart(style(fig), width="stretch")

        fig = px.histogram(d, x="Contract", color="Status", barmode="group",
                           color_discrete_map=STATUS_COLORS,
                           category_orders={"Contract": ["Month-to-month", "One year", "Two year"]},
                           labels={"Contract": "Vertragsart", "count": "Anzahl Kunden"},
                           title="Kündigungen nach Vertragsart")
        right.plotly_chart(style(fig), width="stretch")
        st.info("Kunden mit Monatsvertrag kündigen deutlich häufiger als Kunden mit "
                "Jahresverträgen.")

    # --- Tab 2: Einzelner Kunde ---
    with tab2:
        st.caption("Wähle einen Kunden aus dem Testset und sieh das Risiko.")
        customer = st.selectbox("Kunde", res["ids"].tolist())
        idx = res["ids"][res["ids"] == customer].index[0]
        p = float(res["proba"].loc[idx])
        actual = "hat gekündigt" if res["y_test"].loc[idx] == 1 else "ist geblieben"

        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100,
            number={"suffix": " %", "valueformat": ".1f"},
            title={"text": "Kündigungswahrscheinlichkeit"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1B2430"},
                   "steps": [{"range": [0, 30], "color": "#B7E4C7"},
                             {"range": [30, 60], "color": "#FFE8A3"},
                             {"range": [60, 100], "color": "#F5B7B1"}]},
        ))
        left, right = st.columns([2, 1])
        left.plotly_chart(style(gauge, 300), width="stretch")
        right.metric("Tatsächlich", actual)
        risiko = "hoch" if p >= 0.6 else "mittel" if p >= 0.3 else "niedrig"
        right.metric("Risiko", risiko)

        # Download aller Testkunden mit Vorhersage
        out = pd.DataFrame({
            "customerID": res["ids"],
            "Kuendigungswahrscheinlichkeit": res["proba"].round(4),
            "Tatsaechlich_gekuendigt": res["y_test"],
        }).sort_values("Kuendigungswahrscheinlichkeit", ascending=False)
        st.download_button("⬇️ Alle Vorhersagen als CSV", to_csv(out),
                           "churn_vorhersagen.csv", "text/csv")

    # --- Tab 3: Modellbewertung ---
    with tab3:
        left, right = st.columns(2)

        left.subheader("Confusion Matrix")
        left.caption("Zeilen = Realität, Spalten = Vorhersage.")
        fig = px.imshow(res["cm"], text_auto=True, color_continuous_scale="Blues",
                        x=["Bleibt", "Kündigt"], y=["Bleibt", "Kündigt"],
                        labels=dict(x="Vorhersage", y="Realität"))
        fig.update_coloraxes(showscale=False)
        left.plotly_chart(style(fig, 400), width="stretch")

        right.subheader("Wichtigste Einflussfaktoren")
        right.caption("Rot = erhöht das Kündigungsrisiko, grün = senkt es.")
        coefs = res["coefs"]
        top = pd.concat([coefs.head(8), coefs.tail(8)]).sort_values()
        fig = px.bar(x=top.values, y=top.index, orientation="h",
                     color=top.values > 0,
                     color_discrete_map={True: RED, False: GREEN},
                     labels={"x": "Koeffizient (skalierte Features)", "y": ""})
        fig.update_layout(showlegend=False)
        right.plotly_chart(style(fig, 400), width="stretch")

        st.subheader("Alle Kennzahlen")
        st.dataframe(pd.Series(m).round(3).to_frame("Wert"), width="stretch")


# ---------------------------------------------------------------------------
# Seite 2: Forecast
# ---------------------------------------------------------------------------
def page_forecast():
    st.title("📈 Umsatzprognose (Rossmann)")
    st.markdown(
        "Das Zeitreihenmodell **Prophet** lernt Trend sowie Wochen- und "
        "Jahresmuster der Tagesumsätze und sagt die kommenden Tage voraus."
    )
    sales = get_sales()

    # Einstellungen in der Sidebar
    st.sidebar.header("Einstellungen")
    store = int(st.sidebar.selectbox("Store", sorted(sales["Store"].unique())))
    days = st.sidebar.slider("Prognose-Horizont (Tage)", 7, 60, 30)

    with st.spinner("Prophet wird trainiert ..."):
        fc = get_forecast(store, days)

    hist = sales[sales["Store"] == store]
    last_hist = hist["Date"].max()
    future = fc[(fc["ds"] > last_hist) & (fc["ds"].dt.dayofweek != 6)].copy()  # ohne Sonntage
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        future[col] = future[col].clip(lower=0)

    # --- KPI-Karten ---
    best = future.loc[future["yhat"].idxmax()]
    last30 = hist[hist["Date"] > last_hist - pd.Timedelta(days=30)]["Sales"].mean()
    avg_fc = future["yhat"].mean()
    delta = avg_fc / last30 - 1
    c = st.columns(4)
    kpi(c[0], "Ø Tagesumsatz (Prognose)", f"{avg_fc:,.0f} €".replace(",", "."),
        f"{delta:+.1%} zu letzten 30 Tagen", GREEN if delta >= 0 else RED)
    kpi(c[1], "Summe Prognose", f"{future['yhat'].sum():,.0f} €".replace(",", "."),
        f"nächste {days} Tage", BLUE)
    kpi(c[2], "Stärkster Tag", f"{best['yhat']:,.0f} €".replace(",", "."),
        best["ds"].strftime("%a, %d.%m.%Y"), ORANGE)
    kpi(c[3], "Ø letzte 30 Tage", f"{last30:,.0f} €".replace(",", "."),
        "tatsächlicher Umsatz", GRAY)
    st.write("")

    # --- Zeitraum wählen ---
    min_d, max_d = hist["Date"].min().date(), fc["ds"].max().date()
    default_start = max(min_d, (last_hist - pd.Timedelta(days=120)).date())
    rng = st.date_input("Zeitraum", value=(default_start, max_d),
                        min_value=min_d, max_value=max_d)
    if len(rng) != 2:
        st.info("Bitte Start- und Enddatum wählen.")
        return
    start, end = map(pd.to_datetime, rng)
    h = hist[(hist["Date"] >= start) & (hist["Date"] <= end)]
    f = fc[(fc["ds"] >= start) & (fc["ds"] <= end)]

    # --- Diagramm: Historie (Punkte) + Prognose (Linie + Band) ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f["ds"], y=f["yhat_upper"], mode="lines",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=f["ds"], y=f["yhat_lower"], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(31,111,235,0.15)", name="Unsicherheit",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f["ds"], y=f["yhat"], mode="lines",
                             line=dict(color=BLUE, width=2.5), name="Prognose"))
    fig.add_trace(go.Scatter(x=h["Date"], y=h["Sales"], mode="markers",
                             marker=dict(color="#1B2430", size=5), name="Historisch"))
    fig.add_shape(type="line", x0=last_hist, x1=last_hist, y0=0, y1=1,
                  yref="paper", line=dict(dash="dash", color=GRAY))
    fig.add_annotation(x=last_hist, y=1, yref="paper", text="Heute", showarrow=False,
                       xanchor="left", font=dict(color=GRAY))
    fig.update_layout(yaxis_title="Umsatz (€)", hovermode="x unified")
    st.plotly_chart(style(fig, 450), width="stretch")

    # --- Tabelle + Download ---
    st.subheader("Prognose der nächsten Tage")
    st.caption("Sonntage sind ausgeblendet, da die Läden geschlossen sind.")
    table = future[["ds", "yhat", "yhat_lower", "yhat_upper"]].round(0).rename(columns={
        "ds": "Datum", "yhat": "Prognose (€)",
        "yhat_lower": "Untergrenze (€)", "yhat_upper": "Obergrenze (€)"})
    table["Datum"] = table["Datum"].dt.date
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("⬇️ Prognose als CSV", to_csv(table),
                       f"prognose_store_{store}_{days}tage.csv", "text/csv")


# ---------------------------------------------------------------------------
# Sidebar-Navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Churn & Forecast")
page = st.sidebar.radio("Seite", ["📉 Customer Churn", "📈 Forecast"],
                        label_visibility="collapsed")
st.sidebar.caption("Datenquellen: Telco Customer Churn, Rossmann Store Sales")
st.sidebar.divider()

if page.endswith("Customer Churn"):
    page_churn()
else:
    page_forecast()