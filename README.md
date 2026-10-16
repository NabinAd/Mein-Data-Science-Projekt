# Customer Churn und Umsatzprognose

Data-Science-Projekt mit zwei Anwendungsfällen: Kündigungen bei Telekommunikationskunden analysieren und Tagesumsätze von drei Rossmann-Filialen prognostizieren. Die Arbeit umfasst Datenaufbereitung, explorative Analysen, Modellvergleich und ein interaktives Streamlit-Dashboard.

## Schnellstart für die Prüfung

Voraussetzung: **Python 3.12**. Alle Befehle im Projektordner ausführen.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Unter macOS/Linux mit installiertem Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Die im Terminal angezeigte lokale Adresse öffnen, normalerweise `http://localhost:8501`. Das erste Training kann etwas dauern; weitere Aufrufe werden zwischengespeichert. Die benötigten Eingabedaten liegen bereits im Repository. Die große Rossmann-Datei `train.csv` ist für das Dashboard nicht erforderlich.

`requirements.txt` enthält Python-Pakete; Python selbst wird separat installiert. `.python-version` dokumentiert die gewünschte Python-Version. Die direkten Paketversionen sind festgelegt; ein vollständiger Lock aller indirekten Abhängigkeiten ist nicht enthalten.

## Daten und Quellen

| Datei | Inhalt |
|---|---|
| `data/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv` | 7.043 Kunden, 21 Spalten; 1.869 Kündigungen (26,54 %) |
| `data/rossmann/train_filtered.csv` | 2.826 Tagesdatensätze für Stores 1, 2 und 3, vom 01.01.2013 bis 31.07.2015 |
| `data/rossmann/forecasts.csv` | Gespeicherter Prophet-Export mit 2.434 Zeilen inklusive historischer Modellwerte und Zukunftsprognosen |
| `data/rossmann/train.csv` | 1017210  Datensätze von Rossmann |

Datenquellen: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) und [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data). Die Daten stammen aus externen Quellen; deren Nutzungsbedingungen gelten unabhängig vom Projektcode. Eine Erlaubnis zur Weiterveröffentlichung wird durch diese README nicht erteilt. Die Bedingungen für die öffentliche Bereitstellung der CSV-Dateien sollten vor der Abgabe geprüft werden.

Aus Datenschutz- und Lizenzgründen habe ich die Originaldatensätze nicht in dieses Repository hochgeladen. Sie können über die oben angegebenen Links heruntergeladen werden. Legen Sie die heruntergeladenen Dateien anschließend entsprechend ihrer Datenquelle in einem der folgenden Projektordner ab:

WA_Fn-UseC_-Telco-Customer-Churn.csv (Telco-Datensatz) : data/telco/
train.csv (Rossmann-Datensatz): data/rossmann/

Die mitgelieferten Dateien erlauben das Nachvollziehen der Analyse. Es sind historische Beispieldaten: Die Prognose beginnt nach dem letzten Datenstand von 2015, nicht am heutigen Datum. Die Umsatzachse ist im Projekt als Euro beschriftet; eine verifizierte Währungsangabe ist in den lokalen CSV-Dateien nicht enthalten.

## Aufbau

```text
app.py                              Interaktives Dashboard
src/churn.py                        Datenbereinigung und logistische Regression
src/forecasting.py                  Prophet-Prognose pro Filiale
01_daten_vorbereitung.ipynb          Optional: Originaldaten auf drei Stores filtern
Erste_Diagramm_erstellen.ipynb       Datenüberblick und Diagramme
Chrun_prediction.ipynb              Churn-Modelle und Vergleich
forecasting_mit_Prophet.ipynb        Prophet, zeitlicher Test und CSV-Export
data/telco/                         Telco-Eingabedaten
data/rossmann/                      Gefilterte Umsatzdaten und Prognoseexport
.streamlit/config.toml              Darstellung des Dashboards
requirements.txt                   Python-Abhängigkeiten
```

Die vier PNG-Dateien im Hauptordner zeigen die Churn-Verteilung, Kündigungen nach Vertragsart sowie Umsatzverteilung und Umsatzverlauf. Die bestehenden Notebook-Dateinamen wurden beibehalten.

## Notebooks nachvollziehen

Unter Windows Jupyter mit `.venv\Scripts\python.exe -m notebook` starten, unter macOS/Linux mit `.venv/bin/python -m notebook`.

1. Optional `01_daten_vorbereitung.ipynb`: benötigt zusätzlich die Rossmann-Originaldatei unter `data/rossmann/train.csv`. Diese ist nicht enthalten. Mit dem mitgelieferten gefilterten Datensatz kann dieser Schritt übersprungen werden.
2. `Erste_Diagramm_erstellen.ipynb`: Datenüberblick und vier Diagramme.
3. `Chrun_prediction.ipynb`: logistische Regression, Random Forest und Modellvergleich.
4. `forecasting_mit_Prophet.ipynb`: Prognose, Validierung für Store 1 und Export für alle drei Stores.

Jeweils den Kernel neu starten und alle Zellen von oben nach unten ausführen. Diagramme und `forecasts.csv` werden dabei neu geschrieben. Vorhandene Notebook-Ausgaben dokumentieren frühere Läufe und können von neu berechneten Ergebnissen abweichen.

## Methoden und vorhandene Ergebnisse

### Customer Churn

`TotalCharges` enthält elf Leerwerte, die in Zahlen umgewandelt und durch den Median ersetzt werden. `customerID` ist kein Modellmerkmal. Kategorische Merkmale werden per One-Hot-Encoding umgewandelt. Es folgt ein stratifizierter 80/20-Train/Test-Split mit `random_state=42`. Die Skalierung wird nur am Trainingssatz angepasst.

Das Dashboard verwendet logistische Regression. Das Notebook vergleicht zusätzlich einen Random Forest. Folgende Werte stammen aus den **gespeicherten Notebook-Ausgaben**:

| Kennzahl | Logistische Regression | Random Forest |
|---|---:|---:|
| Accuracy | 80,70 % | 78,64 % |
| Precision | 65,84 % | 62,37 % |
| Recall | 56,68 % | 49,20 % |
| F1-Score | 60,92 % | 55,01 % |
| ROC-AUC | 0,8416 | 0,8251 |

Die logistische Regression schneidet in diesem gespeicherten Vergleich besser ab. Der Recall zeigt jedoch, dass viele tatsächliche Kündiger nicht erkannt werden. Eine Mehrheitsvorhersage „bleibt“ erreicht im gesamten Datensatz bereits rund 73,46 % Accuracy. Deshalb sind Recall, F1 und ROC-AUC zusätzlich wichtig.

### Umsatzprognose

Prophet verwendet Trend, Wochen- und Jahressaisonalität. Trainiert wird je Store auf Tagen mit `Open == 1` und positivem Umsatz; insgesamt sind das 2.344 Zeilen. Der Prognosehorizont im Dashboard liegt zwischen 7 und 60 Kalendertagen. Sonntage werden für Zukunftstabelle und Kennzahlen ausgeblendet und negative Werte dort auf null begrenzt. Die Diagrammlinie enthält dagegen die unveränderten Modellwerte.

Das Notebook hält für Store 1 die letzten 30 Kalendertage zurück und bewertet die darin enthaltenen geöffneten Tage. Gespeichertes Ergebnis: **MAE rund 616 Umsatzeinheiten je bewertetem Tag**. Die ausgegebenen **13,8 %** sind `MAE / mittlerer tatsächlicher Umsatz`, also ein normalisierter MAE und **kein MAPE**. Das Dashboard trainiert anschließend auf der gesamten vorhandenen Historie. Es lädt den gespeicherten Prognoseexport nicht ein.

## Grenzen und nächste Verbesserungen

- Der Median für `TotalCharges` und die Kategorien des Encodings werden derzeit vor dem Split bestimmt. Für eine methodisch saubere Evaluation sollten Imputation und Encoding in eine ausschließlich auf Trainingsdaten angepasste Pipeline verschoben werden.
- Churn-Ergebnisse beruhen auf einem einzelnen Split. Eine Kreuzvalidierung auf den Trainingsdaten, ein unangetasteter Abschlusstest und eine begründete Entscheidungsschwelle fehlen.
- Koeffizienten zeigen Zusammenhänge im Modell, keine kausalen Einflüsse. Die Risikostufen 30/60 % sind Darstellungsgrenzen; der Klassifikator verwendet standardmäßig 50 %. Eine Kalibrierung wurde nicht nachgewiesen.
- Der Prophet-Test umfasst nur Store 1 und ein Zeitfenster. Ein Vergleich mit einer einfachen Wochenprognose und mehrere zeitliche Testfenster für alle drei Stores fehlen.
- Aktionen, Feiertage und zusätzliche Schließtage werden nicht als Prädiktoren berücksichtigt. Die Prognosesumme ist deshalb keine belastbare vollständige Kalender-Umsatzplanung.
- Es handelt sich um ein Lernprojekt; ein nachgewiesener finanzieller Nutzen oder ein Einsatz mit aktuellen Kundendaten ist nicht Bestandteil dieser Arbeit.

## Was gehört ins Repository?

Code, Notebooks, die gezielt ausgewählten CSV-Dateien und Ergebnisgrafiken bleiben für die Prüfung enthalten. Jupyter-Checkpoints, Python-Caches, virtuelle Umgebungen und Zugangsdaten werden ignoriert.

## Technische Prüfung am 16.10.2026

Die Paketinstallation und die Abhängigkeitsprüfung waren unter Python 3.12.10 erfolgreich. Churn-Training und Kennzahlen wurden reproduziert, die Codezellen des Diagramm- und Churn-Notebooks vollständig ausgeführt und die Churn-Seite einschließlich Kundenauswahl mit Streamlits AppTest geprüft. Die vier Notebook-Strukturen und die Python-Syntax sind gültig.
