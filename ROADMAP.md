# Roadmap

Prioritäten nach der technischen Überarbeitung vom 26.10.2026. Offene Punkte sind geplante Arbeit, keine bereits vorhandenen Funktionen.

## Als Nächstes

- [ ] Churn-Pipeline mit stratifizierter Kreuzvalidierung auf Trainingsdaten prüfen; abschließenden Testsatz erst für die Endbewertung verwenden.
- [ ] Entscheidungsschwelle anhand dokumentierter Fehlerkosten auf Validierungsdaten wählen und Precision/Recall vergleichen.
- [ ] Kalibrierung der Churn-Scores prüfen; keine ungetesteten Wahrscheinlichkeitsversprechen.
- [ ] Forecast-Backtests über mehrere Zeitfenster für alle drei Filialen; identische Testtage für Prophet und Wochen-Baseline verwenden.
- [ ] Öffnungszeiten, Feiertage und Aktionen in eine klar definierte Kalender-Umsatzprognose integrieren.

## Weitere technische Verbesserungen

- [ ] Zwei große UI-Seiten bei weiterer Erweiterung in eigene Module auslagern; aktuelle fachliche Logik bleibt unter src/.
- [ ] Optional eine kleine API für Vorhersagen mit Eingabeschema und Integrationstests ergänzen.
- [ ] Optional eine Live-Demo veröffentlichen; Ressourcenbedarf, Datenzugriff und Cache-Verhalten vorher testen.
- [ ] Vollständigen Abhängigkeits-Lock und Installation in weiteren Betriebssystemen prüfen.
- [ ] Eine bewusste Lizenzentscheidung für eigenen Code treffen; Datenquellen haben eigene Nutzungsbedingungen.

## Bereits umgesetzt

- [x] README mit Problem, Anwendungsvorschau, Tech-Stack, Startanleitung, Ergebnissen und Grenzen.
- [x] Churn-Pipeline: Train/Test-Split vor gelernter Vorverarbeitung.
- [x] Mehrheits-Baseline für Churn und Wochen-Baseline im Forecast-Notebook.
- [x] CSV-Validierung und verständliche Anzeige erwartbarer Datenfehler.
- [x] pytest, Streamlit-AppTests, Ruff und GitHub-Actions-Workflow.
- [x] Notebooks in notebooks/, Grafiken in docs/images/, portable Datenpfade.
- [x] Generierte Cache-Dateien und lokale Editor-Einstellungen aus dem versionierten Stand entfernt.
- [x] Direkte Paketversionen festgelegt und Notebook-Abläufe neu ausgeführt.
